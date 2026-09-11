"""Compare installed versions with an explicitly selected pinned profile."""
import argparse,importlib.metadata,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--requirements',type=Path,required=True);a=p.parse_args();errors=[]
    for line in a.requirements.read_text().splitlines():
        if not line.strip() or line.startswith(('#','--')):continue
        name,expected=line.split('==',1)
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:errors.append(name+': missing');continue
        if actual!=expected and not ('+' not in expected and actual.split('+')[0]==expected):errors.append(f'{name}: expected {expected}, got {actual}')
    if sys.version_info[:2]!=(3,12):errors.append('Python 3.12 required for publication checks')
    if errors:raise SystemExit('\n'.join(errors))
    print('ENVIRONMENT_PROFILE_VERIFIED')
if __name__=='__main__':main()
