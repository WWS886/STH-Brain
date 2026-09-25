"""Read and validate the immutable pretraining subject protocol."""
from utils.artifact_schema import schema_matches
from pathlib import Path
from collections import Counter,defaultdict
import csv,json
import hashlib
SCHEMA='timebrain_subject_protocol'

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    return digest.hexdigest()

FILES={'manifest.csv','subjects.json','tasks.json','ppmi_pairs.json','rest_pairs.json'}

def _read(bundle):
    b=Path(bundle);receipt=json.loads((b/'receipt.json').read_text())
    if not schema_matches(receipt.get('schema'),SCHEMA) or set(receipt.get('artifacts',{}))!=FILES:raise ValueError('Wrong/incomplete  split receipt')
    for name,digest in receipt['artifacts'].items():
        if sha(b/name)!=digest:raise ValueError('Split artifact changed: '+name)
    for item in receipt['clinical_sources']:
        if sha(item['source'])!=item['sha256']:raise ValueError('Source workbook changed: '+item['source'])
    with (b/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:scans=list(csv.DictReader(f))
    people=json.loads((b/'subjects.json').read_text())
    return receipt,scans,people

def validate_bundle(manifest,receipt=None):
    manifest=Path(manifest).resolve();b=manifest.parent
    if manifest.name!='manifest.csv' or (receipt and Path(receipt).resolve()!=b/'receipt.json'):raise ValueError('Require manifest and its own receipt')
    rec,scans,people=_read(b);byid={p['person_uid']:p for p in people};by_scan={s['scan_uid']:s for s in scans}
    if len(byid)!=len(people) or len(by_scan)!=len(scans):raise ValueError('Duplicate subject/scan identity')
    if {s['subject_uid'] for s in scans}!=set(byid):raise ValueError('Inventory coverage mismatch')
    if not schema_matches(rec.get('protocol',{}).get('version'),'fixed712'):
        raise ValueError('Pretraining requires the registered fixed 7:1:2 split')
    return validate_fixed(scans,people,b)


def load_manifest(bundle,split):
    if split not in ['train','val']:raise ValueError('Pretraining may only read train/val')
    _rec,scans,_people=_read(bundle)
    return [r for r in scans if r['split']==split]

def validate_fixed(scans,people,b):
    """Validate registered pretraining membership and exclusion of held-out roles."""
    counts=Counter(p['split'] for p in people)
    if len(people)!=12774 or counts!=dict(train=9040,val=905,test=1803,external=620,zero_shot=406):
        raise ValueError('Fixed712 subject totals')
    byid={p['person_uid']:p for p in people}
    role_groups=defaultdict(set)
    for scan in scans:
        person=byid[scan['subject_uid']]
        if scan['split']!=person['split'] or scan['component_uid']!=person['component_uid']:
            raise ValueError('Subject/scan mismatch')
        if int(scan['additional_trim'])!=0 or int(scan['n_roi'])!=200 or int(scan['length'])<=0:
            raise ValueError('Sequence metadata')
        for key in ('subject_uid','component_uid','content_sha256'):
            role_groups[(key,scan[key])].add(scan['split'])
    if any(len(roles)>1 for roles in role_groups.values()):
        raise ValueError('Cross-role identity/content/component leakage')
    return dict(status='pass',schema=SCHEMA,protocol='fixed712',subjects=len(people),scans=len(scans),
                subject_counts=dict(counts),component_exclusive=True,
                held_out_roles_excluded_from_pretraining=True,additional_trim=0,
                manifest_sha256=sha(b/'manifest.csv'),receipt_sha256=sha(b/'receipt.json'))
