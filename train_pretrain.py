#!/usr/bin/env python3
"""STH-Brain eight-GPU pretraining; immutable subject rosters and exact resume."""
import argparse,contextlib,copy,csv,fcntl,json,math,os,random,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from settings.pretrain import load_config,DEFAULT_CONFIG,ROOT,ALGORITHM_VERSION,PretrainObjective
from model.config import STHBrainConfig
from model.brain.pretrain import BrainPretrainer,representation_health
from dataset.pretrain_data import make_dataset,make_collate,CompleteSubjectBatchSampler,SubjectEvaluationSampler,model_inputs
from utils.pretrain_contract import atomic_json,source_manifest,input_contract,sha
from utils.telemetry import Journal
from utils.optimization import ExactCollapseMonitor,clip_gradients,snapshot_parameters,parameter_updates,loss_gradient_diagnostics


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def set_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
def save_checkpoint(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def gpu_uuid_locks(gpu_ids):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    mapping={row.split(',',1)[0].strip():row.split(',',1)[1].strip() for row in output.splitlines() if ',' in row}
    requested=[item.strip() for item in gpu_ids.split(',') if item.strip()]
    known_uuids=set(mapping.values())
    missing=[item for item in requested if item not in mapping and item not in known_uuids]
    if missing:raise ValueError(f'GPU indices or UUIDs are unavailable: {missing}')
    lock_dir=Path(os.environ.get('STH_BRAIN_GPU_LOCK_DIR','/tmp/sth_brain_gpu_locks'));lock_dir.mkdir(parents=True,exist_ok=True)
    handles=[]
    try:
        for uuid in sorted({mapping.get(item,item) for item in requested}):
            handle=(lock_dir/f'{uuid}.lock').open('a')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except Exception:
                handle.close();raise
            handles.append(handle)
    except Exception:
        for handle in handles:handle.close()
        raise RuntimeError(f'Requested GPU set is already owned by another STH-Brain run: {requested}')
    return handles

def _process_identity(pid):
    try:fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return None
    return fields[0],fields[19]

def _descendant_pids(root_pid):
    descendants={};pending=[root_pid]
    while pending:
        parent=pending.pop()
        try:
            children=(Path(f'/proc/{parent}/task/{parent}/children').read_text().split())
        except (FileNotFoundError,ProcessLookupError):
            continue
        for value in children:
            child=int(value)
            if child not in descendants:
                identity=_process_identity(child)
                if identity is not None:
                    descendants[child]=identity[1];pending.append(child)
    return descendants

def _signal_pids(pids,signum):
    items=pids.items() if isinstance(pids,dict) else ((pid,None) for pid in pids)
    for pid,starttime in sorted(items,reverse=True):
        identity=_process_identity(pid)
        if identity is None or (starttime is not None and identity[1]!=starttime):continue
        try:os.kill(pid,signum)
        except (ProcessLookupError,PermissionError):pass

def _wait_pids_exit(pids,timeout=5):
    deadline=time.monotonic()+timeout
    while True:
        live={}
        items=pids.items() if isinstance(pids,dict) else ((pid,None) for pid in pids)
        for pid,starttime in items:
            identity=_process_identity(pid)
            if identity is None or (starttime is not None and identity[1]!=starttime):continue
            if identity[0]!='Z':live[pid]=identity[1]
        if not live:return
        if time.monotonic()>=deadline:
            _signal_pids(live,signal.SIGKILL)
            raise RuntimeError(f'Child descendants did not terminate after SIGKILL: {sorted(live)}')
        time.sleep(.02)

def wait_for_child(process,process_group=False,terminate_timeout=10):
    """Wait while turning catchable termination signals into child cleanup."""
    previous={}
    def interrupted(signum,_frame):
        raise SystemExit(128+signum)
    for signum in (signal.SIGTERM,signal.SIGINT):
        previous[signum]=signal.getsignal(signum);signal.signal(signum,interrupted)
    try:
        return process.wait()
    except BaseException:
        for signum in previous:signal.signal(signum,signal.SIG_IGN)
        group_pid=getattr(process,'pid',None) if process_group else None
        descendants=_descendant_pids(group_pid) if group_pid is not None else set()
        if process.poll() is None:
            if group_pid is not None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except OSError:pass
            else:process.terminate()
            try:
                process.wait(timeout=terminate_timeout)
                _signal_pids(descendants,signal.SIGKILL)
                _wait_pids_exit(descendants)
            except subprocess.TimeoutExpired:
                if group_pid is not None:descendants.update(_descendant_pids(group_pid))
                if group_pid is not None:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except OSError:pass
                else:process.kill()
                _signal_pids(descendants,signal.SIGKILL)
                process.wait()
                _wait_pids_exit(descendants)
        else:
            _signal_pids(descendants,signal.SIGKILL)
            _wait_pids_exit(descendants)
        raise
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)

def loader(dataset,cfg,epoch,rank=0,world=1,train=False,limit=0):
    col=make_collate(cfg,epoch if train else 0,seed=cfg['runtime']['seed']+(0 if train else 10000))
    generator=torch.Generator().manual_seed(cfg['runtime']['seed']+epoch*31+rank)
    kw=dict(num_workers=cfg['optim']['num_workers'],pin_memory=True,collate_fn=col,generator=generator)
    if train:
        sampler=CompleteSubjectBatchSampler(dataset.df.subject_uid.tolist(),cfg['optim']['batch_size']*world,world,rank,cfg['runtime']['seed']);sampler.set_epoch(epoch)
        return DataLoader(dataset,batch_sampler=sampler,**kw)
    sampler=SubjectEvaluationSampler(dataset.df.subject_uid.tolist(),world,rank,limit*cfg['optim']['batch_size'])
    return DataLoader(dataset,batch_size=cfg['optim']['batch_size'],sampler=sampler,**kw)

@torch.no_grad()
def evaluate(model,dataset,cfg,device,limit=0,distributed=False,stage_diagnostics=False):
    model.eval();errors={};embeddings={};full_raw={};full_ln={};source_by_subject={};tr_by_subject={};stage_rows={}
    rank=dist.get_rank() if distributed else 0
    world=dist.get_world_size() if distributed else 1
    error=None
    from utils.stage_health import select_stage_subjects
    stage_subjects=select_stage_subjects(dataset.df,cfg['runtime']['stage_health_max_subjects']) if stage_diagnostics else set()
    layer_rows={}
    try:
        for index,batch in enumerate(loader(dataset,cfg,0,rank,world,limit=limit)):
            inputs=model_inputs(batch,device)
            trace_batch=stage_diagnostics and any(s in stage_subjects for s in inputs['subject_uids'])
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=cfg['optim']['amp'] and device.type=='cuda'):
                out=model(**inputs,objective_distributed=False,evaluation_only=True,stage_diagnostics=trace_batch)
                readouts={'raw':out['student_full_raw_embeddings'],'layernorm':out['student_full_embeddings']}
            for name,values in out.get('stage_vectors',{}).items():
                bucket=stage_rows.setdefault(name,{})
                for subject,value in zip(inputs['subject_uids'],values.cpu()):
                    if subject in stage_subjects:bucket.setdefault(subject,[]).append(value)
            for name,(values,present) in out.get('layer_slots',{}).items():
                bucket=layer_rows.setdefault(name,{})
                for subject,value,mask in zip(inputs['subject_uids'],values.cpu(),present.cpu()):
                    if subject in stage_subjects:bucket.setdefault(subject,[]).append((value.clone(),mask.clone()))
            for s,e,z in zip(inputs['subject_uids'],out['scan_prediction_loss'].cpu(),out['student_embeddings'].cpu()):
                errors.setdefault(s,[]).append(float(e));embeddings.setdefault(s,[]).append(z)
            for subject,tr in zip(inputs['subject_uids'],inputs['tr_seconds'].cpu().tolist()):tr_by_subject.setdefault(subject,[]).append(tr)
            for subject,source in zip(inputs['subject_uids'],inputs['source_names']):
                if subject in source_by_subject and source_by_subject[subject]!=source:raise ValueError('Validation subject spans sources')
                source_by_subject[subject]=source
            for s,raw,ln in zip(inputs['subject_uids'],readouts['raw'].cpu(),readouts['layernorm'].cpu()):
                full_raw.setdefault(s,[]).append(raw);full_ln.setdefault(s,[]).append(ln)
    except Exception as exc:
        error=repr(exc)
    local=dict(error=error,errors=errors,embeddings=embeddings,full_raw=full_raw,full_ln=full_ln,sources=source_by_subject,trs=tr_by_subject,stage_rows=stage_rows,layer_rows=layer_rows)
    if distributed:
        # No collectives inside unequal local loops. CPU tensors serialize once
        # to rank 0; non-root ranks never compute duplicate cohort eigenspectra.
        gathered=[None]*world if rank==0 else None
        dist.gather_object(local,gathered,dst=0)
        if rank!=0:return None
    else:gathered=[local]
    failures=[row['error'] for row in gathered if row['error']]
    if failures:raise RuntimeError('Validation shard failed: '+str(failures))
    errors={};embeddings={};full_raw={};full_ln={};source_by_subject={};tr_by_subject={};stage_rows={};layer_rows={}
    for row in gathered:
        if set(errors).intersection(row['errors']):raise ValueError('Validation subject duplicated across ranks')
        errors.update(row['errors']);embeddings.update(row['embeddings']);tr_by_subject.update(row['trs'])
        for name,items in row['stage_rows'].items():stage_rows.setdefault(name,{}).update(items)
        for name,items in row['layer_rows'].items():layer_rows.setdefault(name,{}).update(items)
        full_raw.update(row['full_raw']);full_ln.update(row['full_ln']);source_by_subject.update(row['sources'])
    if not errors:raise ValueError('Empty validation cohort')
    # Canonical order makes serial/distributed health reduction comparable.
    errors={s:errors[s] for s in sorted(errors)}
    embeddings={s:embeddings[s] for s in errors}
    full_raw={s:full_raw[s] for s in errors};full_ln={s:full_ln[s] for s in errors}
    z=torch.stack([torch.stack(v).mean(0) for v in embeddings.values()])
    result={'val_prediction_loss':float(np.mean([np.mean(v) for v in errors.values()])),'val_subjects':len(errors)}
    source_losses={source:float(np.mean([np.mean(values) for subject,values in errors.items() if source_by_subject[subject]==source])) for source in sorted(set(source_by_subject.values()))}
    result['val_source_macro_prediction_loss']=float(np.mean(list(source_losses.values())))
    result.update({'val_source_'+source+'_prediction_loss':value for source,value in source_losses.items()})
    result.update({'val_student_'+k:v for k,v in representation_health(z).items()})
    # Distinguish masked prediction features from the full-view moment target.
    for name,rows in (('full_raw',full_raw),('full_ln',full_ln)):
        values=torch.stack([torch.stack(v).mean(0) for v in rows.values()])
        result.update({'val_student_'+name+'_'+k:v for k,v in representation_health(values).items()})
    if stage_rows:
        from utils.stage_health import detailed_health
        stage_report={}
        for name,items in stage_rows.items():
            subjects=sorted(items);vectors=torch.stack([torch.stack(items[s]).mean(0) for s in subjects])
            stage_report[name]={'all':detailed_health(vectors),'by_source':{}}
            stage_report[name]['by_tr']={}
            tr_groups=[('le_1_5',lambda t:t<=1.5),('1_5_to_2_5',lambda t:1.5<t<=2.5),('gt_2_5',lambda t:t>2.5)]
            for label,predicate in tr_groups:
                selected=[i for i,s in enumerate(subjects) if predicate(float(np.mean(tr_by_subject[s])))]
                if len(selected)>=2:stage_report[name]['by_tr'][label]=detailed_health(vectors[selected])
            for source in sorted(set(source_by_subject.values())):
                selected=[i for i,s in enumerate(subjects) if source_by_subject[s]==source]
                if len(selected)>=2:stage_report[name]['by_source'][source]=detailed_health(vectors[selected])
        result['stage_health']=stage_report
    if layer_rows:
        from utils.stage_health import build_layer_report
        import hashlib
        panel=dataset.df.loc[dataset.df.subject_uid.isin(stage_subjects.intersection(errors))]
        columns=[key for key in ('subject_uid','scan_uid','content_sha256','dataset') if key in panel]
        panel_records=panel[columns].sort_values(columns).to_dict('records')
        provenance={'panel_sha256':hashlib.sha256(json.dumps(panel_records,sort_keys=True).encode()).hexdigest(),
                    'mask_seed':cfg['runtime']['seed']+10000,'mask_epoch':0,
                    'precision':'bf16' if cfg['optim']['amp'] and device.type=='cuda' else 'fp32',
                    'device_type':device.type,'panel_scan_count':len(panel)}
        result['layer_diagnostics']=build_layer_report(layer_rows,source_by_subject,stage_subjects.intersection(errors),
            provenance=provenance,compression_ratio=cfg['runtime'].get('compression_rank_ratio',.5),
            compression_minimum_rank=cfg['runtime'].get('compression_minimum_reference_rank',.05))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--output-dir','--output_dir',dest='output_dir')
    p.add_argument('--resume',default='auto');p.add_argument('--gpus');p.add_argument('--stop-after',type=int,default=500)
    p.add_argument('--seed',type=int,help='Override the pretraining seed without changing the registered split')
    p.add_argument('--max-train-batches',type=int,default=0);p.add_argument('--max-val-batches',type=int,default=0)
    p.add_argument('--smoke',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--protocol',choices=['formal','exploratory'],default='formal')
    args=p.parse_args()
    from settings.pretrain import resolve_pretrain_runtime
    cfg=resolve_pretrain_runtime(load_config(args.config),seed=args.seed,output_dir=args.output_dir)
    # Validate the new cohort before GPU allocation, child launch or output creation.
    if 'RANK' not in os.environ:
        from dataset.protocol import validate_bundle
        validate_bundle(cfg['data']['manifest'],cfg['data']['split_receipt_path'])
    if args.gpus:cfg['runtime']['gpus']=args.gpus
    if args.smoke:
        cfg['optim']['num_workers']=0
    elif args.max_train_batches or args.max_val_batches:raise ValueError('Limits require --smoke and an isolated output')
    run=Path(cfg['runtime']['output_dir']);gpu_ids=cfg['runtime']['gpus']
    if args.smoke and 'smoke' not in str(run):raise ValueError('Smoke must use a named smoke output directory')
    if args.dry_run:
        print(json.dumps({'config':cfg,'input_contract':input_contract(cfg),'source_sha256':source_manifest()['sha256']},indent=2));return
    if 'RANK' not in os.environ and gpu_ids!='cpu':
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu_ids;env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');env.setdefault('OMP_NUM_THREADS','2');env.setdefault('MKL_NUM_THREADS','2')
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(gpu_ids.split(','))),str(ROOT/'train_pretrain.py'),*sys.argv[1:]]
        locks=gpu_uuid_locks(gpu_ids)
        process=None
        try:
            process=subprocess.Popen(cmd,env=env,cwd=ROOT,start_new_session=True,pass_fds=tuple(handle.fileno() for handle in locks))
            returncode=wait_for_child(process,process_group=True,terminate_timeout=45)
        finally:
            for handle in locks:handle.close()
        if returncode:raise SystemExit(returncode)
        return
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if not args.smoke and world!=cfg['runtime']['expected_world_size']:raise ValueError('Formal training requires exactly eight ranks')
    device=torch.device('cpu' if gpu_ids=='cpu' else f'cuda:{local}')
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(minutes=30))
    lock=None;journal=None
    try:
        setup=[None]
        if rank==0:
            try:
                run.mkdir(parents=True,exist_ok=True);lock=(run/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                source=source_manifest();inputs=input_contract(cfg)
                if not args.smoke:
                    if args.protocol=='exploratory' and 'pilot' not in run.name and 'exploratory' not in str(run):
                        raise ValueError('Exploratory protocol requires an isolated pilot/exploratory output directory')
                resume=None
                if args.resume=='auto':resume=run/'checkpoint_last.pt' if (run/'checkpoint_last.pt').exists() else None
                elif args.resume in ('none','fresh'):
                    if (run/'history.csv').exists() or (run/'checkpoint_last.pt').exists():raise ValueError('Fresh run refuses existing training output')
                elif args.resume.isdigit():resume=run/f'checkpoint_epoch_{int(args.resume):03d}.pt'
                else:resume=Path(args.resume).resolve()
                if resume and not resume.is_file():raise FileNotFoundError(resume)
                # An earlier-epoch branch must have a fresh destination, preserving history.
                if resume and resume.parent!=run and (run/'history.csv').exists():raise ValueError('Branch output is not empty')
                setup[0]={'source':source,'inputs':inputs,'resume':str(resume) if resume else None}
            except Exception as exc:setup[0]={'error':repr(exc)}
        if world>1:dist.broadcast_object_list(setup,src=0)
        if 'error' in setup[0]:raise RuntimeError(setup[0]['error'])
        source=setup[0]['source'];inputs=setup[0]['inputs'];resume=setup[0]['resume']
        seed_all(cfg['runtime']['seed']+rank)
        model=BrainPretrainer(STHBrainConfig(**cfg['model']),PretrainObjective(**cfg['objective'])).to(device)
        optimizer=torch.optim.AdamW([{'params':model.student.parameters(),'lr':cfg['optim']['student_lr'],'name':'student'},{'params':model.predictor.parameters(),'lr':cfg['optim']['predictor_lr'],'name':'predictor'}],weight_decay=cfg['optim']['weight_decay'])
        collapse_monitor=ExactCollapseMonitor(cfg['optim']['exact_collapse_patience'])
        train_ds=make_dataset(cfg,'train');val_ds=make_dataset(cfg,cfg['data']['val_split'])
        limits={'smoke':args.smoke,'protocol':args.protocol,'train_batches':args.max_train_batches,'val_batches':args.max_val_batches,'world_size':world}
        start=0;updates=0;best=float('inf');history=[]
        if resume:
            ck=torch.load(resume,map_location='cpu',weights_only=False)
            if ck.get('algorithm_version')!=ALGORITHM_VERSION or ck['source_manifest']!=source or ck['input_contract']!=inputs or ck['run_limits']!=limits:raise ValueError('Resume algorithm/source/input/topology mismatch')
            def science(c):return {k:v for k,v in c.items() if k!='runtime'}|{'seed':c['runtime']['seed']}
            if science(ck['config'])!=science(cfg):raise ValueError('Resume model/objective/data/schedule mismatch')
            model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer']);start=ck['epoch'];updates=ck['updates'];best=ck['best_prediction_loss']
            collapse_monitor.load_state_dict(ck['exact_collapse_monitor'])
            history=ck['history'];set_rng(ck['rng_by_rank'][rank])
            # Never erase newer epochs by requesting an old checkpoint in-place.
            if (run/'history.csv').exists():
                with (run/'history.csv').open() as stream:rows=list(csv.DictReader(stream))
                if rows and int(rows[-1]['epoch'])>start:raise ValueError('Earlier-epoch resume requires a new output directory')
        wrapped=DDP(model,device_ids=[local] if device.type=='cuda' else None,broadcast_buffers=False,find_unused_parameters=True) if world>1 else model
        # DDP constructor synchronizes parameters; restore after construction too.
        if resume:set_rng(ck['rng_by_rank'][rank])
        if rank==0:
            atomic_json(run/'resolved_config.json',cfg);atomic_json(run/'source_manifest.json',source);atomic_json(run/'input_contract.json',inputs)
            atomic_json(run/'environment_manifest.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'world_size':world,'python':sys.version,'deterministic':True,'tf32':False})
            journal=Journal(run)
            print(f'STH-Brain TS-JEPA | source={source["sha256"]} | resume_epoch={start} | ranks={world} | no FC | output={run}',flush=True)
        batches_per_epoch=len(loader(train_ds,cfg,1,rank,world,True));batches_per_epoch=min(batches_per_epoch,args.max_train_batches) if args.max_train_batches else batches_per_epoch
        steps_per_epoch=math.ceil(batches_per_epoch/cfg['optim']['accumulation_steps']);total_updates=500*steps_per_epoch
        # A fixed train-only subject/mask panel makes gradient snapshots comparable.
        # DataLoader has its own generator; preserve all model RNG state as well.
        diagnostic_data=[]
        if cfg['optim']['gradient_diagnostic_every']:
            saved_rng=rng_state()
            for diagnostic_index,diagnostic_batch in enumerate(loader(train_ds,cfg,1,rank,world,True)):
                diagnostic_data.append(diagnostic_batch)
                if len(diagnostic_data)>=cfg['optim']['gradient_diagnostic_batches']:break
            set_rng(saved_rng)
        stopped=[False]
        def stop_handler(*_):stopped[0]=True
        signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
        for epoch in range(start+1,min(500,args.stop_after)+1):
            timer=time.monotonic();model.train();sums={k:0. for k in ('loss','prediction_loss','roi_prediction_loss','network_prediction_loss','variance_loss','covariance_loss','student_feature_std')};seen=0;optimizer.zero_grad(set_to_none=True)
            dl=loader(train_ds,cfg,epoch,rank,world,True)
            for batch_index,batch in enumerate(dl):
                if batch_index>=batches_per_epoch:break
                group_start=(batch_index//cfg['optim']['accumulation_steps'])*cfg['optim']['accumulation_steps']
                group_size=min(cfg['optim']['accumulation_steps'],batches_per_epoch-group_start)
                sync=(batch_index+1==group_start+group_size)
                for group in optimizer.param_groups:
                    progress=updates/max(1,total_updates-1);warm=cfg['optim']['warmup_epochs']*steps_per_epoch
                    ratio=(updates+1)/warm if updates<warm else cfg['optim']['minimum_lr_ratio']+(1-cfg['optim']['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(updates-warm)/max(1,total_updates-warm)))
                    group['lr']=cfg['optim'][group['name']+'_lr']*ratio
                model.training_progress=updates/max(total_updates,1)
                model.optimizer_update=updates;model.steps_per_epoch=steps_per_epoch
                data=model_inputs(batch,device)
                diagnostic_every=cfg['optim']['gradient_diagnostic_every']
                if batch_index==group_start and diagnostic_every and updates % diagnostic_every==0:
                    panel_index=(updates//diagnostic_every)%len(diagnostic_data)
                    diagnostic=loss_gradient_diagnostics(model,model_inputs(diagnostic_data[panel_index],device),amp=cfg['optim']['amp'] and device.type=='cuda',distributed=world>1)
                    if rank==0:journal.log(f'gradient/{updates}',{'epoch':epoch,'optimizer_step':updates,'panel_index':panel_index,'panel_batches':len(diagnostic_data),**diagnostic})
                context=wrapped.no_sync() if world>1 and not sync else contextlib.nullcontext()
                with context:
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=cfg['optim']['amp'] and device.type=='cuda'):
                        result=wrapped(**data,objective_distributed=world>1,
                            diagnostics=(batch_index % cfg['runtime']['health_every']==0 or batch_index+1==batches_per_epoch))
                    if not torch.isfinite(result['loss']):raise FloatingPointError('Non-finite pretraining loss')
                    if collapse_monitor.observe(result['moment_health']):
                        failure={'status':'screening_failed','reason':'sustained_exact_constant_representation',
                                 'epoch':epoch,'batch_index':batch_index,'optimizer_step':updates,
                                 'moment_health':result['moment_health'],'exact_collapse_monitor':collapse_monitor.state_dict(),
                                 'updated_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source['sha256']}
                        if rank==0:
                            atomic_json(run/'screening_failure.json',failure)
                            atomic_json(run/'status.json',failure)
                            journal.log(f'screening_failure/{epoch}/{batch_index}',failure)
                        raise RuntimeError(f'Exact constant subject representation persisted for {collapse_monitor.patience} consecutive training microbatches')
                    (result['loss']/group_size).backward()
                optimizer_metrics={}
                if sync:
                    optimizer_metrics=clip_gradients(model,cfg['optim'])
                    interval=cfg['optim']['update_diagnostic_every']
                    before=snapshot_parameters(model) if rank==0 and interval and updates % interval==0 else None
                    optimizer.step()
                    if before is not None:optimizer_metrics.update(parameter_updates(model,before))
                    optimizer.zero_grad(set_to_none=True);updates+=1
                    momentum=cfg['model']['ema_final']-(cfg['model']['ema_final']-cfg['model']['ema_base'])*.5*(1+math.cos(math.pi*updates/total_updates))
                    model.ema_update(momentum)
                n=result['subject_count'];seen+=n
                for key in sums:sums[key]+=float(result[key].detach())*n
                if rank==0 and (batch_index%cfg['runtime']['log_every']==0 or batch_index+1==batches_per_epoch):
                    payload={'epoch':epoch,'micro_step':(epoch-1)*batches_per_epoch+batch_index+1,'optimizer_step':updates,**{'step/'+k:float(result[k].detach()) for k in sums},'step/subjects':n,'step/accumulation_group_size':group_size,
                             **{'moment_health/'+key:value for key,value in result['moment_health'].items()},
                             'moment_health/consecutive_exact_constant':collapse_monitor.consecutive}
                    payload.update({f'branch_health/{branch}/{key}':value for branch,health in result.get('branch_health',{}).items() for key,value in health.items()})
                    payload.update(optimizer_metrics)
                    payload['microbatch_diagnostics_scope']='rank0_local_microbatch; moment_support_is_global_unique_subjects'
                    payload.update({key:result[key] for key in ('normalization_diagnostics','target_group_diagnostics')})
                    payload['prediction_group_report']=result['prediction_group_report']
                    payload['teacher_target_diagnostics']=result['teacher_target_diagnostics']
                    payload['roi_routing']=result['roi_routing_diagnostics']
                    payload['step/subject_logdet_loss']=float(result['subject_logdet_loss'].detach())
                    payload['mask_diagnostics']=result['mask_diagnostics']
                    payload['native_support_diagnostics']=result['native_support_diagnostics']
                    payload['identity_moment_evidence']=dict(zip(('roi','network'),result['identity_moment_evidence']))
                    payload['identity_moment_scope']=result['identity_moment_scope']
                    payload['regularizer_view_evidence']=result['regularizer_view_evidence']
                    payload['loss_protocol']=result['loss_protocol']
                    payload['covariance_actual_weight']=result['covariance_actual_weight']
                    payload['weighted_objective_terms']={key:float(value.detach()) for key,value in result['objective_terms'].items()}
                    payload['step/huber_unweighted']=float(result['huber_loss'])
                    payload['step/cosine_unweighted']=float(result['cosine_loss'])
                    journal.log(f'micro/{epoch}/{batch_index}',payload)
                    if batch_index%10==0:print(f'epoch {epoch:03d}/500 batch {batch_index+1}/{batches_per_epoch} loss={float(result["loss"].detach()):.6f} pred={float(result["prediction_loss"].detach()):.6f} var={float(result["variance_loss"].detach()):.6f} logdet={float(result["subject_logdet_loss"].detach()):.6f}',flush=True)
            train_seconds=time.monotonic()-timer
            validation=[None]
            try:validation[0]=evaluate(model,val_ds,cfg,device,args.max_val_batches,distributed=world>1,
                stage_diagnostics=epoch%cfg['runtime']['stage_health_every_epochs']==0)
            except Exception as exc:validation[0]={'error':repr(exc)}
            if world>1:dist.broadcast_object_list(validation,src=0)
            if 'error' in validation[0]:raise RuntimeError(validation[0]['error'])
            state=rng_state();rngs=[None]*world
            if world>1:dist.all_gather_object(rngs,state)
            else:rngs=[state]
            save_status=[None]
            if rank==0:
                try:
                    stage_health=validation[0].pop('stage_health',None)
                    if stage_health is not None:
                        atomic_json(run/f'stage_health/epoch{epoch:03d}.json',{'epoch':epoch,'scope':'pretraining_diagnostics','source_sha256':source['sha256'],'metrics':stage_health})
                    layer_report=validation[0].pop('layer_diagnostics',None)
                    row={'epoch':epoch,**{'train_'+k:v/seen for k,v in sums.items()},**validation[0],'train_subject_exposures':seen,'updates':updates,'train_seconds':train_seconds,'epoch_total_seconds':time.monotonic()-timer}
                    row['train_subject_logdet_loss']=row['train_covariance_loss']
                    history.append(row);improved=row['val_prediction_loss']<best;best=min(best,row['val_prediction_loss'])
                    ck={'algorithm_version':ALGORITHM_VERSION,'epoch':epoch,'updates':updates,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':{'updates':updates,'total_updates':total_updates},'config':copy.deepcopy(cfg),'architecture_fingerprint':model.architecture_fingerprint(),'source_manifest':source,'input_contract':inputs,'rng_by_rank':rngs,'run_limits':limits,'history':history,'best_prediction_loss':best,'exact_collapse_monitor':collapse_monitor.state_dict(),'roi_qc_receipt_path':cfg['data']['qc_receipt_path'],'roi_qc_receipt_sha256':cfg['data']['qc_receipt_sha256']}
                    save_checkpoint(run/'checkpoint_last.pt',ck)
                    if epoch%cfg['runtime']['save_every']==0 or epoch in cfg['runtime']['checkpoint_epochs'] or args.smoke:save_checkpoint(run/f'checkpoint_epoch_{epoch:03d}.pt',ck)
                    diagnostic_checkpoint=run/f'checkpoint_epoch_{epoch:03d}.pt'
                    if not diagnostic_checkpoint.exists():diagnostic_checkpoint=run/'checkpoint_last.pt'
                    checkpoint_digest=sha(diagnostic_checkpoint) if layer_report is not None else None
                    if layer_report is not None:
                        from utils.stage_health import annotate_compression_persistence
                        interval=cfg['runtime']['stage_health_every_epochs']
                        previous_path=run/f'layer_diagnostics/epoch{epoch-interval:03d}.json'
                        previous=json.loads(previous_path.read_text()) if previous_path.is_file() else None
                        provenance={**layer_report['provenance'],'source_sha256':source['sha256'],
                                    'checkpoint_sha256':checkpoint_digest,'checkpoint_epoch':epoch}
                        layer_report['provenance']=provenance
                        for entry in layer_report['layers'].values():entry['provenance']=dict(provenance)
                        annotate_compression_persistence(layer_report,previous,epoch,interval=interval,
                            patience=cfg['runtime'].get('compression_patience',3))
                        atomic_json(run/f'layer_diagnostics/epoch{epoch:03d}.json',{'source_sha256':source['sha256'],**layer_report})
                        print('LAYER_DIAGNOSTICS '+json.dumps({'epoch':epoch,'subjects':layer_report['subject_count'],
                            'layers':layer_report['layer_count'],'suspects':layer_report['suspects'],
                            'compression_needs_review':[k for k,v in layer_report['compression'].items() if v['needs_review']]}),flush=True)
                    if improved:
                        save_checkpoint(run/'checkpoint_best.pt',ck)
                        save_checkpoint(run/'checkpoint_prediction_best.pt',ck)
                        atomic_json(run/'prediction_best.json',{'epoch':epoch,'criterion':'val_prediction_loss','value':best})
                    temp=run/'history.csv.tmp'
                    with temp.open('w') as f:
                        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
                    os.replace(temp,run/'history.csv');journal.log(f'epoch/{epoch}',{'epoch':epoch,**{'epoch_metrics/'+k:v for k,v in row.items()}})
                    atomic_json(run/'status.json',{'status':'training','epoch':epoch,'target_epochs':500,'pid':os.getpid(),'updated_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source['sha256']})
                    print('EPOCH_COMPLETE '+json.dumps(row),flush=True)
                    stopped[0] |= (run/'STOP_AFTER_EPOCH').exists()
                    save_status[0]={'stop':stopped[0]}
                except Exception as exc:save_status[0]={'error':repr(exc)}
            if world>1:dist.broadcast_object_list(save_status,src=0)
            if 'error' in save_status[0]:raise RuntimeError(save_status[0]['error'])
            stop=torch.tensor(int(stopped[0] or save_status[0]['stop']),device=device)
            if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if stop.item():break
        if rank==0:
            from utils.training_state import finalize_training_state
            finalize_training_state(run,history[-1]['epoch'] if history else start,cfg,source,
                smoke=args.smoke)
    finally:
        if journal:journal.close()
        if lock:lock.close()
        if dist.is_initialized():dist.destroy_process_group()


if __name__ == "__main__":
    main()
