"""Read-only paginated audit of actual GitHub PR/review activity."""
import argparse,json,subprocess
from pathlib import Path
MEMBERS=['GeminusF','Nicat-Agayev','optim00s','milanakarimova','Jeyhunaa']
def api(endpoint):
    raw=subprocess.check_output(['gh','api','--paginate','--slurp',endpoint],text=True)
    return [item for page in json.loads(raw) for item in page]
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='GeminusF/sac-qutab');p.add_argument('--output',type=Path,default=Path('output/github-audit.json'));a=p.parse_args()
    counts={m:{'merged_prs':0,'substantive_commit_shas':set(),'approved_prs':set()} for m in MEMBERS};prs=[]
    for pr in api(f'repos/{a.repo}/pulls?state=all&per_page=100'):
        if not pr['head']['ref'].startswith('codex/publication-'):continue
        if not pr.get('merged_at'):continue
        author=next((m for m in MEMBERS if m.lower()==pr['user']['login'].lower()),None)
        if author:counts[author]['merged_prs']+=1
        commits=api(f'repos/{a.repo}/pulls/{pr["number"]}/commits?per_page=100')
        for commit in commits:
            user=(commit.get('author') or {}).get('login','')
            member=next((m for m in MEMBERS if m.lower()==user.lower()),None)
            if member and len(commit['parents'])==1:counts[member]['substantive_commit_shas'].add(commit['sha'])
        for review in api(f'repos/{a.repo}/pulls/{pr["number"]}/reviews?per_page=100'):
            member=next((m for m in MEMBERS if m.lower()==review['user']['login'].lower()),None)
            if member and review['state']=='APPROVED' and review['commit_id']==pr['head']['sha']:counts[member]['approved_prs'].add(pr['number'])
        prs.append({'number':pr['number'],'author':author,'head_sha':pr['head']['sha'],'merge_sha':pr['merge_commit_sha']})
    for item in counts.values():
        for key in ('substantive_commit_shas','approved_prs'):item[key]=sorted(item[key])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'repo':a.repo,'counts':counts,'prs':prs},indent=2)+'\n')
    print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
