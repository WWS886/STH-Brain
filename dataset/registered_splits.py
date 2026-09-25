"""Registered STH-Brain subjects and untrimmed audited ROI sequences.

The manifest is immutable. Train/val caches are derived artifacts; only train/val
sequences can be constructed by this pretraining data interface.
"""
from utils.artifact_schema import schema_matches
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataset.protocol import load_manifest

ROOT = Path(__file__).resolve().parents[1]
from utils.paths import resolve_paths
DEFAULT_BUNDLE = Path(resolve_paths('${STH_BRAIN_SPLIT_DIR}'))
DEFAULT_CACHE = Path(resolve_paths('${STH_BRAIN_CACHE_DIR}'))
INTERFACE = 'registered_subject_tasks'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read_txt(row):
    # Check and parse the same bytes. Every stored frame is retained.
    import io
    payload=Path(row['seq_path']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=row['content_sha256']:
        raise ValueError(f'TXT changed: {row["scan_uid"]}')
    x=np.loadtxt(io.BytesIO(payload),dtype=np.float32,ndmin=2)
    if x.shape!=(int(row['length']),200) or not np.isfinite(x).all() or int(row['additional_trim'])!=0:
        raise ValueError(f'TXT shape/finite/trim mismatch: {row["scan_uid"]}')
    return x


class RegisteredDataset(Dataset):
    def __init__(self, manifest, split, *, cache_dir=DEFAULT_CACHE):
        self.bundle=Path(manifest).resolve().parent
        self.cache_dir=Path(cache_dir); self.n_roi=200
        roles=[split] if isinstance(split,str) else list(split)
        if not roles or len(set(roles))!=len(roles): raise ValueError('Distinct explicit roles required')
        if not set(roles)<={'train','val'}:
            raise ValueError('Pretraining can only read train/val')
        rows=[]
        for role in roles:
            rows.extend(load_manifest(self.bundle,role))
        if not rows: raise ValueError(f'Empty registered dataset: {roles}')
        self.df=pd.DataFrame(rows)
        self.df['tr']=pd.to_numeric(self.df['tr'],errors='raise')
        for key in ('length','n_roi','valid_rois','additional_trim'):
            self.df[key]=pd.to_numeric(self.df[key],errors='raise').astype(int)
        if not self.df.tr.gt(0).all():
            raise ValueError('Invalid registered TR')
        receipt=self.cache_dir/'sequence_cache_receipt.json'
        self.cache=json.loads(receipt.read_text()) if receipt.is_file() else None
        if self.cache and (not schema_matches(self.cache.get('schema'),INTERFACE) or self.cache['manifest_sha256']!=sha(manifest)):
            raise ValueError('Cache belongs to another registered manifest')

    def __len__(self): return len(self.df)

    def __getitem__(self,index):
        r=self.df.iloc[index].to_dict(); key=r['scan_uid']
        if self.cache and r.get('pretrain_split',r['split']) in {'train','val'}:
            e=self.cache['entries'].get(key)
            if not e or e['source_sha256']!=r['content_sha256']: raise ValueError(f'Cache identity mismatch: {key}')
            path=self.cache_dir/e['file']
            if sha(path)!=e['sha256']: raise ValueError(f'Cache changed: {key}')
            x=np.load(path,allow_pickle=False)
        else: x=read_txt(r)
        if x.shape!=(int(r['length']),200) or not np.isfinite(x).all(): raise ValueError(f'Invalid sequence: {key}')
        mask=x.std(axis=0)>=1e-6
        if not mask.any() or int(mask.sum())!=int(r['valid_rois']): raise ValueError(f'ROI QC drift: {key}')
        t=torch.arange(len(x),dtype=torch.float32)*float(r['tr'])
        return {'x':torch.from_numpy(x.copy()),'timestamps_seconds':t,
                'length':int(r['length']),'tr':float(r['tr']),
                'subject_uid':r['subject_uid'],'scan_uid':key,'dataset':r['dataset'],
                'roi_valid_mask':torch.from_numpy(mask)}
