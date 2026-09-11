"""Verify a supplied evidence/resource manifest without downloads or mutation."""
import argparse,json,hashlib
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    for rel,record in json.loads(a.manifest.read_text())['files'].items():
        path=(a.root/rel).resolve()
        if not path.is_relative_to(a.root.resolve()):raise ValueError('Manifest traversal')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if path.stat().st_size!=record['bytes'] or h.hexdigest()!=record['sha256']:raise ValueError('Resource mismatch: '+rel)
    print('RESOURCE_HASHES_VERIFIED')
if __name__=='__main__':main()
