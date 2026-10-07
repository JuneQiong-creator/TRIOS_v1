"""Check release payload bytes, never compute scientific quantities."""
from pathlib import Path
import argparse,csv,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
def verify(root=ROOT):
    checked=0
    listed=set()
    with (root/'provenance/PAYLOAD_SHA256.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            p=(root/r['file']).resolve()
            assert p.is_relative_to(root.resolve()),'Manifest traversal'
            assert p.is_file(),r['file']
            b=p.read_bytes()
            assert len(b)==int(r['bytes']) and hashlib.sha256(b).hexdigest()==r['sha256'],r['file']
            listed.add(r['file'])
            checked+=1
    excluded={'.git','.venv','__pycache__','outputs','build','dist'}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not any(x in excluded or x.endswith('.egg-info') for x in p.relative_to(root).parts) and p.suffix not in {'.pyc','.log'} and p.name!='.DS_Store'}
    assert actual==listed|{'provenance/PAYLOAD_SHA256.csv'}, {'unlisted':sorted(actual-listed-{'provenance/PAYLOAD_SHA256.csv'}),'missing':sorted(listed-actual)}
    return dict(status='PASS',files=checked,scope='Listed release payload bytes only; manifest excludes itself and generated outputs')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    print(json.dumps(verify(a.root.resolve()),indent=2))
