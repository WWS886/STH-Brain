"""Portable paths for the registered STH-Brain split; no dependency on an STH-Brain cache."""
import copy,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def data_root(root):
    return Path(os.environ.get('STH_BRAIN_DATA_ROOT',root)).expanduser().resolve()

def resolve_paths(value,root=ROOT):
    root=Path(os.environ.get('STH_BRAIN_ROOT',root)).expanduser().resolve()
    storage=data_root(root)
    bundle=Path(os.environ.get('STH_BRAIN_SPLIT_DIR',root/'dataset/splits')).expanduser().resolve()
    tokens={
        '${STH_BRAIN_ROOT}':str(root),'${STH_BRAIN_DATA_ROOT}':str(storage),
        '${STH_BRAIN_OUTPUT_ROOT}':str(Path(os.environ.get('STH_BRAIN_OUTPUT_ROOT',root/'outputs')).expanduser().resolve()),
        '${STH_BRAIN_SPLIT_DIR}':str(bundle),
        '${STH_BRAIN_CACHE_DIR}':str(Path(os.environ.get('STH_BRAIN_CACHE_DIR',storage/'cache/pretrain')).expanduser().resolve()),
        '${STH_BRAIN_MANIFEST}':str(Path(os.environ.get('STH_BRAIN_MANIFEST',bundle/'manifest.csv')).expanduser().resolve()),
        '${STH_BRAIN_SPLIT_RECEIPT}':str(Path(os.environ.get('STH_BRAIN_SPLIT_RECEIPT',bundle/'receipt.json')).expanduser().resolve()),
    }
    def visit(item):
        if isinstance(item,dict): return {k:visit(v) for k,v in item.items()}
        if isinstance(item,list): return [visit(v) for v in item]
        if isinstance(item,str):
            for token,replacement in tokens.items(): item=item.replace(token,replacement)
            if '${STH_BRAIN_' in item: raise ValueError(f'Unknown STH-Brain path token: {item}')
        return item
    return visit(copy.deepcopy(value))
