"""Bind registered clinical splits, derived caches and train-only statistics."""
from utils.artifact_schema import schema_matches
from pathlib import Path
import json
from dataset.registered_splits import INTERFACE, sha
from dataset.protocol import validate_bundle


def input_contract(cfg,full_cache=False):
    d=cfg['data']
    if d.get('interface')!=INTERFACE or d['split_col']!='split' or d['train_split']!='train' or d['val_split']!='val':
        raise ValueError('Wrong registered data interface/roles')
    result={'clinical_table_membership':validate_bundle(d['manifest'],d['split_receipt_path'])}
    for key in ('manifest','qc_receipt_path','split_receipt_path','sequence_cache_receipt_path'):
        p=Path(d[key]); digest=sha(p); expected=d.get(key.replace('_path','')+'_sha256')
        if not expected or digest!=expected: raise ValueError('Input SHA mismatch or not prepared: '+key)
        result[key]={'path':str(p),'sha256':digest}
    receipt=json.loads(Path(d['sequence_cache_receipt_path']).read_text())
    if (not schema_matches(receipt['schema'],INTERFACE) or receipt['manifest_sha256']!=sha(d['manifest'])
        or receipt['split_receipt_sha256']!=sha(d['split_receipt_path'])
        or receipt['roles']!=['train','val'] or receipt['additional_trim']!=0):
        raise ValueError('Cache provenance mismatch')
    import pandas as pd
    frame=pd.read_csv(d['manifest'],dtype=str).query('split in ["train", "val"]')
    if set(receipt['entries'])!=set(frame.scan_uid): raise ValueError('Cache roster mismatch')
    root=Path(d['sequence_cache_receipt_path']).parent
    if Path(d['sequence_cache_dir']).resolve()!=root/'roi_sequence_npy': raise ValueError('Cache directory mismatch')
    for r in frame.to_dict('records'):
        entry=receipt['entries'][r['scan_uid']]
        if entry['source_sha256']!=r['content_sha256'] or entry['shape']!=[int(r['length']),200]:
            raise ValueError('Cache entry/source mismatch')
        p=root/entry['file']
        if not p.is_relative_to(root) or not p.is_file() or (full_cache and sha(p)!=entry['sha256']):
            raise ValueError(f'Missing/changed cache: {p}')
    result['cache']={'schema':INTERFACE,'entries':len(receipt['entries']),'additional_trim':0}
    if cfg['model'].get('signal_normalization') in {'train_roi_frozen','visible_scan_roi_zscore'}:
        from utils.normalization import load_normalization_stats,validate_normalization_provenance
        m=cfg['model']; stats=load_normalization_stats(m['normalization_stats_path'],m['normalization_stats_sha256'])
        result['normalization_stats']={'sha256':m['normalization_stats_sha256'],**validate_normalization_provenance(stats,cfg)}
        if stats['subject_count']!=frame.query('split=="train"').subject_uid.nunique(): raise ValueError('Wrong normalization subject count')
    result['fc_artifacts_required']=False
    return result
