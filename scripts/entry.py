"""Launch one pretraining seed with immutable inputs and isolated outputs."""
from pathlib import Path
import argparse,csv,os,shlex,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from settings.pretrain import load_config,resolve_pretrain_runtime,DEFAULT_CONFIG


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='entry',required=True)
    pt=sub.add_parser('pretrain',help='One seed; subject-balanced pretraining and validation')
    pt.add_argument('--config',default=str(DEFAULT_CONFIG))
    pt.add_argument('--resume',default='auto')
    pt.add_argument('--output-dir')
    pt.add_argument('--gpus')
    pt.add_argument('--seed',type=int)
    pt.add_argument('--stop-after',type=int,default=500)
    pt.add_argument('--background',action='store_true')
    pt.add_argument('--foreground',action='store_true')
    pt.add_argument('--dry-run',action='store_true')
    pt.add_argument('--prepare-only',action='store_true')
    pt.add_argument('--smoke',action='store_true')
    pt.add_argument('--max-train-batches',type=int,default=0)
    pt.add_argument('--max-val-batches',type=int,default=0)
    pt.add_argument('--protocol',choices=['formal','exploratory'],default='formal')
    a=p.parse_args()
    cfg=resolve_pretrain_runtime(load_config(a.config),seed=a.seed,output_dir=a.output_dir)
    run=Path(cfg['runtime']['output_dir']);resume=a.resume
    if resume.isdigit():
        ck=run/f'checkpoint_epoch_{int(resume):03d}.pt'
        # Selecting an older completed epoch branches; never truncates history.
        if (run/'checkpoint_last.pt').exists():
            import csv
            history=run/'history.csv'
            if history.exists():
                with history.open() as stream:rows=list(csv.DictReader(stream))
            else:rows=[]
            if rows and int(rows[-1]['epoch'])>int(resume):run=run.parent/(run.name+f'_resume_epoch{int(resume):03d}');resume=str(ck)
    if not 1<=a.stop_after<=500:p.error('--stop-after must be in 1..500')
    if (a.max_train_batches or a.max_val_batches) and not a.smoke:p.error('batch limits require --smoke')
    if a.smoke and (a.max_train_batches not in (1,2) or a.max_val_batches not in (1,2)):p.error('--smoke requires train/val batch limits in {1,2}')
    command=[sys.executable,'-u',str(ROOT/'train_pretrain.py'),'--config',str(Path(a.config).resolve()),'--resume',resume,'--output-dir',str(run),'--stop-after',str(a.stop_after),'--protocol',a.protocol]
    if a.smoke:command+=['--smoke','--max-train-batches',str(a.max_train_batches),'--max-val-batches',str(a.max_val_batches)]
    if a.seed is not None:command+=['--seed',str(a.seed)]
    if a.gpus:command+=['--gpus',a.gpus]
    print(shlex.join(command),flush=True)
    if a.dry_run:
        subprocess.run(command+['--dry-run'],cwd=ROOT,check=True);return
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare.py'),'--config',str(Path(a.config).resolve())],cwd=ROOT,check=True)
    if a.prepare_only:return
    if a.background and not a.foreground:
        run.mkdir(parents=True,exist_ok=True)
        # The trainer owns a process lock; a second command cannot train twice.
        with (run/'master.log').open('a') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
        (run/'launcher.pid').write_text(str(proc.pid)+'\n');print(f'Background PID={proc.pid}; log={run / "master.log"}',flush=True);return
    os.execvpe(command[0],command,os.environ)

if __name__=='__main__':main()
