"""Create local full evidence archives. Publishing requires redistribution review."""
import argparse,json,tarfile
from pathlib import Path
from reproduce import sha
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.output.resolve().is_relative_to(a.source.resolve()):raise ValueError('Output must be outside immutable source')
    files=[]
    for directory in ['evaluations','detectors','runs','statistics','freezes','review','data','counterfactuals','report_artifacts']:
        files.extend(p for p in (a.source/directory).rglob('*') if p.is_file() and 'cache' not in p.parts and not p.name.startswith('wandb') and p.suffix not in ['.log'])
    manifest={'schema':'sac-full-evidence-v1','publication_status':'LOCAL_ONLY_PENDING_REDISTRIBUTION_REVIEW','files':{},'archives':[]}
    import gzip
    class Parts:
        def __init__(self):self.i=0;self.f=None;self.n=0
        def write(self,data):
            total=len(data)
            while data:
                if self.f is None:self.path=a.output/f'full.tar.gz.part{self.i:03d}';self.f=self.path.open('wb');self.n=0
                count=min(len(data),512*1024*1024-self.n);self.f.write(data[:count]);data=data[count:];self.n+=count
                if self.n==512*1024*1024:self.close_part()
            return total
        def flush(self):
            if self.f:self.f.flush()
        def close_part(self):
            if self.f:
                self.f.close();manifest['archives'].append({'name':self.path.name,'bytes':self.n,'sha256':sha(self.path)});self.i+=1;self.f=None
    parts=Parts()
    with gzip.GzipFile(fileobj=parts,mode='wb',compresslevel=3,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|') as tar:
            for f in sorted(files):
                rel=f.relative_to(a.source).as_posix();manifest['files'][rel]={'bytes':f.stat().st_size,'sha256':sha(f)};tar.add(f,arcname=rel,recursive=False)
    parts.close_part();i=parts.i
    (a.output/'full-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'FULL_EVIDENCE_PACKAGED: {len(files)} files, {i} parts; local only')
if __name__=='__main__':main()
