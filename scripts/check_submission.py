"""Check only repository content; runtime evidence remains outside Git."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'github_automation_workflow','report_materials','.omp','_handoff','data','weights','all_artifacts','artifacts','tmp','output','migration-output'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');ap.add_argument('--complete',action='store_true');args=ap.parse_args()
    git=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True)
    paths=[ROOT/p for p in git.stdout.decode().split('\0') if p] if git.returncode==0 else [p for p in ROOT.rglob('*') if p.is_file() and not set(p.relative_to(ROOT).parts)&{'tmp','output','__pycache__','.git'}]
    errors=[]
    for p in paths:
        rel=p.relative_to(ROOT)
        if rel.parts[0] in EXCLUDED: errors.append('Excluded path: '+str(rel))
        if p.stat().st_size>=95*1024*1024: errors.append('Large Git file: '+str(rel))
        if p.suffix in ('.py','.md','.yml','.yaml','.toml','.txt','.tex'):
            text=p.read_text(encoding='utf-8')
            if re.search(r'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)',text):errors.append('Possible credential: '+str(rel))
    for name in ['requirements.txt','src/sac_qutab/cli.py','configs/sac-campaign-v1.yaml']:
        if not (ROOT/name).is_file():errors.append('Missing '+name)
    manifest=ROOT/'report/audit/canonical_assets.json'
    if manifest.exists():
        for rel,digest in json.loads(manifest.read_text()).items():
            if not (ROOT/'report'/rel).is_file() or sha(ROOT/'report'/rel)!=digest:errors.append('Canonical asset mismatch: '+rel)
    if args.final or args.complete:
        for name in ['README.md','report/report.pdf','report/report.tex','presentation/presentation.pdf','contribution_report.pdf','.github/review_template.md','docs/evidence-manifest.json']:
            if not (ROOT/name).is_file():errors.append('Missing '+name)
        signoff=ROOT/'docs/contribution_signoff.json'
        if args.final and (not signoff.exists() or not json.loads(signoff.read_text()).get('all_members_agreed')): errors.append('Contribution assignment awaits all-member sign-off')
        if not (ROOT/'report/audit/pdf_qa.json').exists() or json.loads((ROOT/'report/audit/pdf_qa.json').read_text())['pdf_sha256']!=sha(ROOT/'report/report.pdf'):errors.append('PDF QA hash mismatch')
    release_manifest=ROOT/'docs/release-manifest.json'
    if (args.final or args.complete) and release_manifest.exists():
        for rel,expected in json.loads(release_manifest.read_text())['files'].items():
            p=ROOT/rel
            if not p.is_file() or sha(p)!=expected:errors.append('Release file mismatch: '+rel)
    if errors: raise SystemExit('\n'.join(errors))
    print(f'PACKAGE_CHECK_PASSED: {len(paths)} source files; final={args.final}')
if __name__=='__main__':main()
