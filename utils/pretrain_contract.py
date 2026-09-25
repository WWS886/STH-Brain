"""Exact source/config/input contracts for the sequence-only STH-Brain experiment."""
from pathlib import Path
import hashlib,json,os
from settings.pretrain import ROOT


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def source_manifest():
    files=set(ROOT.glob('*.py'))|set(ROOT.glob('*.sh'))
    for directory in ('model','settings','dataset','utils','scripts','configs'):
        files.update(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py','.sh','.yaml','.yml') and p.is_file())
    files.add(ROOT/'dataset/Node_CC200.node')
    rows={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    return {'files':rows,'sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}

def input_contract(cfg,full_cache=False):
    from utils.registered_contract import input_contract as registered_contract
    return registered_contract(cfg,full_cache)
