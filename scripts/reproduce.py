"""Replay headline estimates from authenticated saved predictions. No training/inference."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, math, shutil, statistics, tarfile, urllib.request

ROOT=Path(__file__).resolve().parents[1]
MODELS=('resnet50','vit_small_patch16_224')
RUNS=tuple(f'{m}-seed{s}' for m in MODELS for s in (17,29,43))
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def require(ok,message):
    if not ok: raise ValueError(message)
def safe_path(root,name):
    rel=PurePosixPath(name)
    require(bool(name) and not rel.is_absolute() and not any(x in ('..','') for x in rel.parts) and ':' not in name and '\\' not in name,'Unsafe archive path')
    target=(Path(root)/name).resolve()
    require(target.is_relative_to(Path(root).resolve()),'Path escapes evidence root')
    return target
def unpack(archive,dest,manifest):
    expected=manifest['files'];seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for entry in tar:
            require(entry.isfile() and entry.name in expected and entry.name not in seen,'Unexpected archive member: '+entry.name)
            require(entry.size==expected[entry.name]['bytes'],'Archive member size mismatch')
            p=safe_path(dest,entry.name);p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(entry) as src,p.open('wb') as out: shutil.copyfileobj(src,out)
            require(sha(p)==expected[entry.name]['sha256'],'Extracted hash mismatch')
            seen.add(entry.name)
    require(seen==set(expected),'Incomplete evidence archive')
def verify(root,manifest):
    for rel,item in manifest['files'].items():
        p=safe_path(root,rel)
        require(p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Evidence mismatch: '+rel)
def fetch(manifest,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    for item in manifest['archives']:
        name=item['name']; p=safe_path(dest,name)
        if not p.exists() or sha(p)!=item['sha256']:
            url=manifest['release_url']+'/'+name
            with urllib.request.urlopen(url,timeout=60) as src,p.with_suffix('.download').open('wb') as out: shutil.copyfileobj(src,out)
            require(sha(p.with_suffix('.download'))==item['sha256'],'Downloaded archive hash mismatch')
            p.with_suffix('.download').replace(p)
    combined=dest/'headline.tar.gz'
    with combined.open('wb') as out:
        for item in manifest['archives']:
            with (dest/item['name']).open('rb') as src: shutil.copyfileobj(src,out)
    unpack(combined,dest/'data',manifest)
    return dest/'data'
def replay(root):
    import numpy as np
    from scipy.special import expit
    from sklearn.metrics import average_precision_score
    root=Path(root);protocol=read(root/'freezes/experiment.json');protocol_sha=sha(root/'freezes/experiment.json')
    require(set(protocol['checkpoint_sha256'])==set(RUNS),'Expected exactly six frozen runs')
    results=[]
    for run in RUNS:
        d=root/'evaluations'/run/'final_test'; ix=read(d/'evaluation_index.json');prov=read(d/'provenance.json')
        require(prov['protocol_sha256']==protocol_sha,'Protocol mismatch')
        for field,name in [('pairs_sha256','pair_metrics.jsonl')]:
            # Evaluation indexes use pair_metrics_sha256; primary statistics use pairs_sha256.
            require(sha(d/name)==ix['pair_metrics_sha256'],'Pair hash does not match frozen index')
        rows=[json.loads(line) for line in (d/'pair_metrics.jsonl').open(encoding='utf-8')]
        require(len(rows)==48600,'Unexpected pair count')
        byid={r['pair_id']:r for r in rows};require(len(byid)==len(rows),'Duplicate pairs')
        clean={}
        for r in rows:
            require(r['role']=='final_test' and r['condition_selected'] and r['audit_qualified'],'Invalid final population')
            value=(r['label'],r['original_probabilities'])
            require(r['source_id'] not in clean or clean[r['source_id']]==value,'Inconsistent clean predictions')
            clean[r['source_id']]=value
        require(len(clean)==4050,'Unexpected clean population')
        acc=statistics.mean(int(np.argmax(p)==label) for label,p in clean.values())*100
        severe=[r for r in rows if r['family']=='resolution' and r['severity']=='severe']
        require(len(severe)==4050,'Unexpected severe-resolution population')
        severe_acc=statistics.mean(int(np.argmax(r['transformed_probabilities'])==r['label']) for r in severe)*100
        det=read(d/'detector_metrics.json'); fitted=read(root/'detectors'/f'{run}.json');pop=det['population_rows']
        require(det['protocol_sha256']==protocol_sha and fitted['fit_split']=='calibration','Detector provenance mismatch')
        require(sha(root/'detectors'/f'{run}.json')==protocol['detector_sha256'][run],'Fitted detector hash mismatch')
        expected={r['pair_id'] for r in rows if r['clean_correct'] and r['severity']!='sham'}
        require(len(pop)==len(expected) and {p['pair_id'] for p in pop}==expected,'Detector population mismatch')
        y=np.array([p['target'] for p in pop]);require(all(p['target']==byid[p['pair_id']]['counterfactual_failure'] for p in pop),'Detector label mismatch')
        aps={}
        for score in ('output_only','output_plus_representation'):
            model=fitted['models'][score]
            x=np.array([[byid[p['pair_id']][f] for f in model['features']] for p in pop])
            pred=expit(np.clip(((x-model['mean'])/model['scale'])@np.asarray(model['coefficient'])+model['intercept'],-60,60))
            require(np.allclose(pred,[p['scores'][score] for p in pop],rtol=2e-7,atol=2e-7),'Detector arithmetic replay differs')
            aps[score]=float(average_precision_score(y,pred))
        results.append({'run':run,'clean_accuracy_pct':acc,'severe_resolution_accuracy_pct':severe_acc,'AP_gain':aps['output_plus_representation']-aps['output_only']})
        print('Replayed '+run,flush=True)
    headline={}
    for model in MODELS:
        rr=[r for r in results if r['run'].startswith(model+'-seed')]
        headline[model]={k:statistics.mean(r[k] for r in rr) for k in ('clean_accuracy_pct','severe_resolution_accuracy_pct','AP_gain')}
        headline[model]['clean_accuracy_sd_pp']=statistics.stdev(r['clean_accuracy_pct'] for r in rr)
        headline[model]['AP_gain_SD']=statistics.stdev(r['AP_gain'] for r in rr)
    return {'protocol_sha256':protocol_sha,'runs':results,'headlines':headline,'scope':'Saved prediction and coefficient replay; no training, neural inference, or bootstrap interval re-estimation.'}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['headline'],default='headline');p.add_argument('--evidence',type=Path);p.add_argument('--output',type=Path,default=ROOT/'output/headline.json');args=p.parse_args()
    manifest=read(ROOT/'docs/evidence-manifest.json')
    evidence=args.evidence or fetch(manifest,ROOT/'all_artifacts/headline')
    verify(evidence,manifest);result=replay(evidence)
    for model,fields in manifest['expected_headlines'].items():
        for key,value in fields.items(): require(math.isclose(result['headlines'][model][key],value,rel_tol=2e-7,abs_tol=2e-7),f'Headline mismatch: {model}/{key}')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('HEADLINE_REPRODUCED: '+str(args.output))
if __name__=='__main__': main()
