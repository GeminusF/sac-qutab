import os
"""Read-only audit of frozen SAC-QUTAB evidence. Writes only report/audit.

No training, model loading, fitting, or inference. Run with Python 3.12 and numpy,
scipy, scikit-learn. Original experiment artifacts are never changed.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
A = Path(os.environ.get('SAC_EVIDENCE_ROOT', str(ROOT / 'all_artifacts')))
OUT = ROOT / 'report/audit'
OUT.mkdir(parents=True, exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def close(a,b, msg, tol=2e-7):
    if not np.allclose(a,b,rtol=tol,atol=tol): raise AssertionError((msg,a,b))
def write(name, data): (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
protocol=read(A/'freezes/experiment.json'); ps=sha(A/'freezes/experiment.json')
stats=read(A/'statistics/primary.json'); index=read(A/'report_artifacts/final/index.json')
assert stats['protocol_sha256']==index['protocol_sha256']==ps
split=read(A/'data/splits.json')
assert sha(A/'data/manifest.jsonl')==protocol['manifest_sha256']
assert sha(A/'data/splits.json')==protocol['split_sha256']
assert Counter(split['sample_roles'].values())==split['counts']
for role,h in protocol['pair_manifest_sha256'].items(): assert sha(A/f'counterfactuals/{role}.jsonl')==h
sources={s['run']:s for s in stats['sources']}
audit={'protocol_sha256':ps,'statistics_sha256':sha(A/'statistics/primary.json'),
       'split':{k:v for k,v in split.items() if k!='sample_roles'},'runs':[],
       'historical_report_index_anomaly':[], 'method':'direct file hashing and independent recomputation from saved outputs; no inference'}
summaries=[]; condition_rows=[]; detector_rows=[]; class_rows=[]; relationships=[]
ledger=[]
for run in sorted(protocol['checkpoint_sha256']):
    print('AUDITING',run,flush=True)
    d=A/'evaluations'/run/'final_test'; m=read(d/'metrics.json'); ix=read(d/'evaluation_index.json'); prov=read(d/'provenance.json')
    assert ix['protocol_sha256']==prov['protocol_sha256']==ps
    mapping={'metrics_sha256':'metrics.json','provenance_sha256':'provenance.json','pair_metrics_sha256':'pair_metrics.jsonl','representations_sha256':'representations.npz'}
    hashes={}
    for field,file in mapping.items():
        h=sha(d/file); assert h==ix[field], (run,field); hashes[file]=h
        sf='pairs_sha256' if field=='pair_metrics_sha256' else field
        assert sources[run][sf]==h,(run,'statistics source',sf)
    for field,file in [('evaluation_index_sha256','evaluation_index.json'),('detector_sha256','detector_metrics.json')]:
        h=sha(d/file); assert h==sources[run][field]; hashes[file]=h
    for field,p in [('checkpoint_sha256',A/'runs'/run/'checkpoints/best.pt'),('detector_sha256',A/'detectors'/f'{run}.json'),('control_ecdf_sha256',A/'evaluations'/run/'validation/control_ecdf.json')]:
        assert sha(p)==protocol[field][run],(run,field)
    resolved=read(A/'runs'/run/'resolved_run.json')
    # config_digest uses compact sorted canonical JSON.
    canonical=json.dumps(resolved,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==protocol['resolved_run_sha256'][run]
    assert prov['checkpoint_sha256']==protocol['checkpoint_sha256'][run]
    old=next(s for s in index['sources'] if '/'+run+'/' in s['directory'].replace('\\','/'))
    if old['representations_sha256']!=hashes['representations.npz']:
        audit['historical_report_index_anomaly'].append({'run':run,'reported':old['representations_sha256'],'actual':hashes['representations.npz']})
    with (d/'pair_metrics.jsonl').open(encoding='utf-8') as f: rows=[json.loads(l) for l in f]
    assert len(rows)==split['counts']['final_test']*12
    by_condition=defaultdict(list); clean={}; byid={}
    for r in rows:
        assert r['role']=='final_test' and r['condition_selected'] and r['audit_qualified']
        assert r['pair_id'] not in byid
        byid[r['pair_id']]=r; by_condition[(r['family'],r['severity'])].append(r)
        clean.setdefault(r['source_id'],r)
    assert len(clean)==split['counts']['final_test']
    y=np.array([r['label'] for r in clean.values()]); p=np.array([r['original_probabilities'] for r in clean.values()]); pred=p.argmax(1)
    close((pred==y).mean(),m['clean']['accuracy'],run+' clean accuracy')
    close(f1_score(y,pred,average='macro'),m['clean']['macro_f1'],run+' clean F1')
    close(np.square(p-np.eye(p.shape[1])[y]).sum(1).mean(),m['clean_calibration']['brier'],run+' Brier')
    close(-np.log(np.clip(p[np.arange(len(y)),y],np.finfo(float).tiny,1)).mean(),m['clean_calibration']['nll'],run+' NLL')
    confidence=p.max(1); correct=pred==y
    assignments=np.clip(np.searchsorted(np.linspace(0,1,16),confidence,side='right')-1,0,14)
    ece=0.0
    for bin_id in range(15):
        mask=assignments==bin_id
        if mask.any(): ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    close(ece,m['clean_calibration']['ece']['ece'],run+' ECE')
    model,seed=run.rsplit('-seed',1); seed=int(seed)
    training=read(A/'runs'/run/'summary.json')
    summaries.append({'run':run,'model':model,'seed':seed,'count':len(y),'accuracy':m['clean']['accuracy'],'macro_f1':m['clean']['macro_f1'],'ece':m['clean_calibration']['ece']['ece'],'brier':m['clean_calibration']['brier'],'nll':m['clean_calibration']['nll'],'best_epoch':training['best_epoch'],'validation_macro_f1':training['best_validation_macro_f1'],'wall_seconds':training['wall_seconds'],'training_device':training['device']['name']})
    for (family,severity),rr in by_condition.items():
        cell=m['conditions'][family+':'+severity]
        record={'run':run,'model':model,'seed':seed,'family':family,'severity':severity,'count':len(rr)}
        for target,field in [('label_consistency','label_consistency'),('transformed_accuracy','transformed_correct'),('js_mean','js_divergence'),('normalized_instability_mean','normalized_representation_instability'),('cosine_mean_within_model','cosine_similarity')]:
            v=float(np.mean([r[field] for r in rr])); close(v,cell[target],run+family+severity+target); record[target]=v
        record['failure_count']=sum(r['counterfactual_failure'] for r in rr)
        record['clean_correct_count']=sum(r['clean_correct'] for r in rr)
        close(record['failure_count'],cell['failure_count'],'failure count',0)
        condition_rows.append(record)
        class_names=protocol['model_contracts'][run]['config']['data']['class_names']
        for cls,cc in cell['class_cells'].items():
            cr=[r for r in rr if r['label']==class_names.index(cls)]
            assert len(cr)==cc['count']
            for target,field in [('label_consistency','label_consistency'),('transformed_accuracy','transformed_correct'),('normalized_instability_mean','normalized_representation_instability')]:
                close(np.mean([r[field] for r in cr]),cc[target],run+cls+target)
            class_rows.append({'run':run,'model':model,'seed':seed,'family':family,'severity':severity,'class_name':cls,**cc})
    # Recompute all simple primary paired effects, independently of report CSVs.
    for s in stats['primary_per_seed']:
        if s['model']!=model or s['seed']!=seed: continue
        fields={'rq1_label_inconsistency_severe_minus_mild':('label_consistency',-1), 'rq1_js_severe_minus_mild':('js_divergence',1),'rq2_normalized_instability_severe_minus_mild':('normalized_representation_instability',1)}
        if s['contrast'] in fields:
            field,sign=fields[s['contrast']]; family=s['family']
            mild={r['source_id']:r for r in by_condition[family,'mild']}
            severe={r['source_id']:r for r in by_condition[family,'severe']}
            values=[sign*(severe[i][field]-mild[i][field]) for i in sorted(mild)]
            close(np.mean(values),s['effect'],run+s['contrast'])
    detector=read(d/'detector_metrics.json'); pop=detector['population_rows']
    expected={r['pair_id'] for r in rows if r['clean_correct'] and r['severity']!='sham'}
    assert {r['pair_id'] for r in pop}==expected and len(pop)==len(expected)
    assert detector['protocol_sha256']==ps
    yy=np.array([r['target'] for r in pop]); assert all(r['target']==byid[r['pair_id']]['counterfactual_failure'] for r in pop)
    fitted=read(A/'detectors'/f'{run}.json')
    assert fitted['fit_split']=='calibration'
    for score in pop[0]['scores']:
        scores=np.array([r['scores'][score] for r in pop])
        if score in fitted['models']:
            from scipy.special import expit
            fitted_model=fitted['models'][score]
            features=np.array([[byid[r['pair_id']][f] for f in fitted_model['features']] for r in pop])
            logits=((features-fitted_model['mean'])/fitted_model['scale'])@np.asarray(fitted_model['coefficient'])+fitted_model['intercept']
            close(expit(np.clip(logits,-60,60)),scores,run+score+' saved coefficient replay')
        else:
            field={'confidence_only':'transformed_uncertainty','entropy_only':'transformed_entropy','representation_only':'normalized_representation_instability'}[score]
            close([byid[r['pair_id']][field] for r in pop],scores,run+score+' raw feature')
        ap=average_precision_score(yy,scores); auc=roc_auc_score(yy,scores)
        stored=[s for s in stats['rq3_absolute_clustered'] if s['model']==model and s['seed']==seed and s['score']==score]
        for s in stored: close(ap if s['metric']=='average_precision' else auc,s['estimate'],run+score+s['metric'])
        detector_rows.append({'run':run,'model':model,'seed':seed,'score':score,'count':len(pop),'positives':int(yy.sum()),'prevalence':float(yy.mean()),'AP':ap,'AUROC':auc})
    # Validate NPZ row alignment and feature-derived metrics without model inference.
    ecdf=np.array(read(A/'evaluations'/run/'validation/control_ecdf.json')['values'])
    with np.load(d/'representations.npz',allow_pickle=False) as z:
        ids=z['pair_ids']; assert ids.tolist()==[r['pair_id'] for r in rows]
        original=z['original']; transformed=z['transformed']; control=z['negative_control']
        assert original.shape==transformed.shape==control.shape
        for start in range(0,len(rows),512):
            end=min(len(rows),start+512); x=original[start:end].astype(float); t=transformed[start:end].astype(float)
            co=(x*t).sum(1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(t,axis=1))
            close(co,[r['cosine_similarity'] for r in rows[start:end]],run+' NPZ cosine',1e-6)
            # The original ECDF query was float32; preserve that rounding.
            xx=original[start:end]; tt=transformed[start:end]
            query=1-(xx*tt).sum(1)/(np.linalg.norm(xx,axis=1)*np.linalg.norm(tt,axis=1))
            normalized=np.searchsorted(ecdf,query,side='right')/len(ecdf)
            close(normalized,[r['normalized_representation_instability'] for r in rows[start:end]],run+' ECDF',1e-6)
        shape=list(original.shape)
    del original,transformed,control
    audit['runs'].append({'run':run,'hashes':hashes,'pair_count':len(rows),'source_count':len(clean),'source_group_count':len({r['source_group_id'] for r in rows}),'representation_shape':shape,'rq3_count':len(pop),'rq3_positives':int(yy.sum()),'verified':True})
    for family in ['resolution','spectral_appearance','texture']:
        rr=[r for r in rows if r['family']==family and r['severity']!='sham']
        # Descriptive pooled within-run correlation; not an inferential test.
        from scipy.stats import spearmanr
        corr=spearmanr([r['js_divergence'] for r in rr],[r['normalized_representation_instability'] for r in rr]).statistic
        relationships.append({'run':run,'family':family,'spearman_js_ecdf':float(corr),'count':len(rr),'analysis':'post-hoc descriptive, correlated severity rows'})
    del rows,byid,by_condition,clean,pop,detector
write('source_audit.json',audit); write('clean.json',summaries); write('conditions.json',condition_rows); write('classes.json',class_rows); write('detectors.json',detector_rows); write('relationships.json',relationships)
write('primary_statistics.json',stats)
assert len(audit['historical_report_index_anomaly'])==5
print('SOURCE_AUDIT_PASSED: all six evaluations, source hashes, checkpoint hashes, pair/NPZ alignment, metrics and detector score metrics verified.',flush=True)
