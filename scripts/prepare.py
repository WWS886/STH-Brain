"""Build verified STH-Brain train/val caches and train-only statistics, then bind YAMLs.

Idempotent: existing artifacts must verify, never overwrite a changed cache.
No test/external/zero-shot TXT is opened and no training process is launched.
"""
from pathlib import Path
import argparse, json, sys, os
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from dataset.registered_splits import RegisteredDataset,INTERFACE,read_txt,sha
from dataset.protocol import load_manifest,validate_bundle
from utils.paths import resolve_paths
from utils.pretrain_contract import atomic_json
from utils.normalization import fit_train_roi_statistics
from settings.pretrain import DEFAULT_CONFIG


def verify_existing_bindings(cfg):
    """Never re-sign an already bound artifact after corruption or manual edits."""
    d=cfg['data'];bindings=[]
    for key in ('manifest','qc_receipt_path','split_receipt_path','sequence_cache_receipt_path'):
        bindings.append((d[key],d.get(key.replace('_path','')+'_sha256')))
    m=cfg['model']
    if m.get('normalization_stats_path'):
        bindings.append((m['normalization_stats_path'],m.get('normalization_stats_sha256')))
    for path,expected in bindings:
        if expected and expected!='NOT_PREPARED' and Path(path).exists() and sha(path)!=expected:
            raise ValueError(f'Already bound artifact changed; refusing to re-sign: {path}')


def prepare(config,full_cache=False):
    cfg=resolve_paths(yaml.safe_load(Path(config).read_text())); d=cfg['data']
    if d.get('interface')!=INTERFACE: raise ValueError('Preparation requires registered STH-Brain configuration')
    if not Path(d['manifest']).is_file():
        raise FileNotFoundError('Provide the registered split using STH_BRAIN_SPLIT_DIR before preparation')
    verify_existing_bindings(cfg)
    validate_bundle(d['manifest'],d['split_receipt_path'])
    # Fully prepared inputs are read and verified without rewriting signed bytes.
    required=[d[k] for k in ('manifest','qc_receipt_path','split_receipt_path','sequence_cache_receipt_path')]
    required.append(cfg['model']['normalization_stats_path'])
    if all(Path(p).is_file() for p in required):
        from settings.pretrain import load_config
        from utils.pretrain_contract import input_contract
        report=input_contract(load_config(config),full_cache)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        return report
    out=Path(d['sequence_cache_receipt_path']).parent; out.mkdir(parents=True,exist_ok=True)
    # Prevent simultaneous preparation jobs from binding incomplete artifacts.
    import fcntl
    with (out/'.prepare.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _prepare(cfg,config,full_cache)


def _prepare(cfg,config,full_cache):
    d=cfg['data']; bundle=Path(d['manifest']).parent
    out=Path(d['sequence_cache_receipt_path']).parent
    rows=load_manifest(bundle,'train')+load_manifest(bundle,'val')
    cache=Path(d['sequence_cache_dir']);cache.mkdir(parents=True,exist_ok=True)
    entries={};qc=[]
    old=json.loads(Path(d['sequence_cache_receipt_path']).read_text()) if Path(d['sequence_cache_receipt_path']).exists() else None
    for i,r in enumerate(rows):
        p=cache/(r['content_sha256']+'.npy')
        e=old['entries'].get(r['scan_uid']) if old else None
        if e:
            if e['source_sha256']!=r['content_sha256'] or sha(p)!=e['sha256'] or sha(r['seq_path'])!=r['content_sha256']:
                raise ValueError(f'Changed cache/source; do not overwrite: {r["scan_uid"]}')
            x=np.load(p,allow_pickle=False)
        else:
            x=read_txt(r)
            if p.exists():
                if not np.array_equal(np.load(p,allow_pickle=False),x): raise ValueError(f'Conflicting cache: {p}')
            else:
                temp=p.with_suffix('.npy.tmp')
                with temp.open('wb') as f:np.save(f,x,allow_pickle=False)
                os.replace(temp,p)
        mask=x.std(axis=0)>=1e-6
        if x.shape!=(int(r['length']),200) or not np.isfinite(x).all() or int(mask.sum())!=int(r['valid_rois']) or not mask.any():
            raise ValueError(f'QC/shape mismatch: {r["scan_uid"]}')
        entries[r['scan_uid']]={'file':str(p.relative_to(out)),'sha256':sha(p),'source_sha256':r['content_sha256'],'shape':list(x.shape)}
        qc.append({'scan_uid':r['scan_uid'],'split':r['split'],'source_sha256':r['content_sha256'],
                   'length':len(x),'roi_valid_mask':''.join('1' if v else '0' for v in mask),'additional_trim':0})
        if (i+1)%500==0:print(f'[prepare] {i+1}/{len(rows)} train/val TXT verified',flush=True)
    receipt={'schema':INTERFACE,'manifest_sha256':sha(d['manifest']),
        'split_receipt_sha256':sha(d['split_receipt_path']),'roles':['train','val'],'additional_trim':0,'entries':entries}
    def immutable_json(path,value):
        path=Path(path)
        if path.exists():
            if json.loads(path.read_text())!=value:raise ValueError(f'Existing artifact differs: {path}')
        else:atomic_json(path,value)
    immutable_json(d['sequence_cache_receipt_path'],receipt)
    immutable_json(d['qc_receipt_path'],{'schema':INTERFACE,'scans':qc})
    stats_path=out/'roi_train_statistics.json'
    if not stats_path.exists():
        print('[prepare] Fitting robust subject-balanced statistics on train only',flush=True)
        ds=RegisteredDataset(d['manifest'],'train',cache_dir=out)
        stats=fit_train_roi_statistics(ds,cfg)
        atomic_json(stats_path,stats)
    # Bind every active pretraining configuration to the same immutable split/cache.
    # No write when values already agree (safe repeated preflight).
    paths=set(ROOT.glob('configs/pretrain*.yaml'))|{Path(config).resolve()}
    for path in sorted(paths):
        raw=yaml.safe_load(path.read_text())
        if raw.get('data',{}).get('interface')!=INTERFACE:continue
        resolved=resolve_paths(raw)
        if Path(resolved['data']['manifest'])!=Path(d['manifest']) or Path(resolved['data']['sequence_cache_receipt_path'])!=Path(d['sequence_cache_receipt_path']):continue
        for key in ('manifest','qc_receipt_path','split_receipt_path','sequence_cache_receipt_path'):
            raw['data'][key.replace('_path','')+'_sha256']=sha(d[key])
        if raw['model']['signal_normalization'] in {'train_roi_frozen','visible_scan_roi_zscore'}:
            raw['model']['normalization_stats_sha256']=sha(stats_path)
        text=yaml.safe_dump(raw,sort_keys=False,allow_unicode=True)
        if path.read_text()!=text:path.write_text(text)
    from settings.pretrain import load_config
    from utils.pretrain_contract import input_contract
    report=input_contract(load_config(config),full_cache)
    atomic_json(out/'preparation_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--full-cache',action='store_true')
    a=p.parse_args();prepare(a.config,a.full_cache)
