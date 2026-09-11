import os
"""Create publication assets from the audited, saved experiment outputs only."""
from pathlib import Path
import json, csv, hashlib, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]; A=Path(os.environ.get('SAC_EVIDENCE_ROOT', str(R.parent/'all_artifacts'))); D=R/'audit'
for sub in ['figures','tables','supplementary']: (R/sub).mkdir(exist_ok=True)
APPROVED_FRAMEWORK_SHA256='fb052ea348c38f35c014df85d24e93bbbc305d71101ec52a83752757ac257b71'
framework_pdf=R/'figures/framework.pdf'
framework_png=R/'figures/framework.png'
if not framework_pdf.is_file() or hashlib.sha256(framework_pdf.read_bytes()).hexdigest()!=APPROVED_FRAMEWORK_SHA256:
    raise RuntimeError('Approved Figure 1 PDF is missing or has changed; restore report/figures/framework.pdf')
if not framework_png.is_file():
    raise RuntimeError('Figure 1 preview is missing; restore report/figures/framework.png')
def read(n): return json.loads((D/n).read_text())
c=read('clean.json'); q=read('conditions.json'); d=read('detectors.json'); cl=read('classes.json'); stats=read('primary_statistics.json'); audit=read('source_audit.json')
M=['resnet50','vit_small_patch16_224']; names=dict(zip(M,['ResNet-50','ViT-Small/16'])); colors=dict(zip(M,['#2864A0','#C27A23'])); markers=dict(zip(M,['o','s']))
families=['spectral_appearance','texture','resolution']; fn=['RGB gain','Amplitude mixing','Resolution']; sevs=['mild','moderate','severe']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.20,'pdf.fonttype':42})
def save(fig,n):
    fig.savefig(R/'figures'/f'{n}.pdf',bbox_inches='tight'); fig.savefig(R/'figures'/f'{n}.png',dpi=160,bbox_inches='tight'); plt.close(fig)
def pm(v,scale=1,digits=3):
    a=np.array(v)*scale
    return f'{a.mean():.{digits}f} $\\pm$ {a.std(ddof=1):.{digits}f}'
def tab(n,headers,rows,spec=None):
    spec=spec or 'l'+'r'*(len(headers)-1)
    text='\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
    text+=''.join(' & '.join(str(x) for x in row)+' \\\\\n' for row in rows)+'\\bottomrule\n\\end{tabular}\n'
    (R/'tables'/f'{n}.tex').write_text(text,encoding='utf-8')
    with (R/'tables'/f'{n}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(headers); w.writerows(rows)

tab('split',['Partition','Images','Groups'],[[role.replace('_',' ').title(),f"{audit['split']['counts'][role]:,}",f"{audit['split']['group_counts'][role]:,}"] for role in ['train','validation','calibration','final_test']])
tab('clean',['Model','Acc. (\\%)','Macro-F1 (\\%)','ECE','Brier','NLL'],[[names[m]]+[pm([r[k] for r in c if r['model']==m],100 if k in ['accuracy','macro_f1'] else 1,2 if k in ['accuracy','macro_f1'] else 4) for k in ['accuracy','macro_f1','ece','brier','nll']] for m in M])
rows=[]
for m in M:
    for f,label in zip(families,fn):
        rr=[r for r in stats['primary_seed_summaries'] if r['model']==m and r['family']==f]
        get=lambda contrast: next(r for r in rr if r['contrast']==contrast)
        row=[names[m],label]
        for contrast in ['rq1_label_inconsistency_severe_minus_mild','rq1_js_severe_minus_mild','rq2_normalized_instability_severe_minus_mild']:
            s=get(contrast); scale=100 if 'label' in contrast else 1
            row.append(f"{s['mean']*scale:.4f} $\\pm$ {s['sd']*scale:.4f}")
        rows.append(row)
tab('contrasts',['Model','Family','$\\Delta$ inconsistency (pp)','$\\Delta$ JS (nats)','$\\Delta$ ECDF'],rows,'llrrr')
rows=[]
score_names={'confidence_only':'Confidence','entropy_only':'Entropy','output_only':'Output only','output_plus_representation':'Output + representation','representation_only':'Representation only'}
for m in M:
    for score in score_names:
        rr=[r for r in d if r['model']==m and r['score']==score]
        rows.append([names[m],score_names[score],pm([r['AP'] for r in rr]),pm([r['AUROC'] for r in rr])])
    for metric in ['average_precision','auroc']:
        s=next(r for r in stats['primary_seed_summaries'] if r['model']==m and r['contrast']=='rq3_nested_detector_delta_'+metric)
        rows.append([names[m],'$\\Delta$ '+('AP' if metric=='average_precision' else 'AUROC'),f"{s['mean']:.3f} $\\pm$ {s['sd']:.3f}" if metric=='average_precision' else '--',f"{s['mean']:.3f} $\\pm$ {s['sd']:.3f}" if metric=='auroc' else '--'])
tab('detectors',['Model','Score / paired gain','AP','AUROC'],rows,'llrr')
tab('per_seed_clean',['Model','Seed','Acc. (\\%)','F1 (\\%)','ECE','Brier','NLL','Best epoch'],[[names[r['model']],r['seed'],f"{100*r['accuracy']:.3f}",f"{100*r['macro_f1']:.3f}",f"{r['ece']:.4f}",f"{r['brier']:.4f}",f"{r['nll']:.4f}",r['best_epoch']+1] for r in c])
tab('population',['Model','Seed','Pairs','Failures','Prevalence (\\%)','Clean-correct sources'],[[names[r['model']],r['seed'],f"{r['count']:,}",f"{r['positives']:,}",f"{100*r['prevalence']:.2f}",r['count']//9] for r in d if r['score']=='output_only'])
rows=[]
for r in stats['primary_per_seed']:
    if r['contrast'].startswith('rq3_nested'):
        rows.append([names[r['model']],r['seed'],'AP' if r['contrast'].endswith('precision') else 'AUROC',f"{r['effect']:.4f}",f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"])
tab('delta_ci',['Model','Seed','Metric','Gain','95\\% clustered CI'],rows,'llrrl')
rows=[]
for m in M:
    for f,label in zip(families,fn):
        for sev in sevs:
            rr=[r for r in q if r['model']==m and r['family']==f and r['severity']==sev]
            rows.append([names[m],label,sev,pm([r['transformed_accuracy'] for r in rr],100,2),pm([r['label_consistency'] for r in rr],100,2),pm([r['cosine_mean_within_model'] for r in rr],1,4),pm([r['normalized_instability_mean'] for r in rr],1,5)])
tab('full_conditions',['Model','Family','Severity','Accuracy (\\%)','Consistency (\\%)','Raw cosine','ECDF'],rows,'lllrrrr')
classes=list(json.loads((A/'freezes/experiment.json').read_text())['model_contracts']['resnet50-seed17']['config']['data']['class_names'])
tab('class_counts',['Class','Train','Validation','Calibration','Final test'],[[cls]+[audit['split']['class_counts'][role][i] for role in ['train','validation','calibration','final_test']] for i,cls in enumerate(classes)])

# Figure contracts are recorded in audit/chart_map.md. All severity points are
# discrete frozen doses, not a smooth or temporally interpolated trend.
fig,axs=plt.subplots(2,3,figsize=(7.16,4.6),layout='constrained')
for j,(f,label) in enumerate(zip(families,fn)):
    for i,(key,ylabel) in enumerate([('transformed_accuracy','Accuracy (%)'),('js_mean','JS divergence (nats)')]):
        ax=axs[i,j]
        for m in M:
            vals=[[r[key]*(100 if i==0 else 1) for r in q if r['model']==m and r['family']==f and r['severity']==s] for s in sevs]
            ax.errorbar(range(3),np.mean(vals,1),yerr=np.std(vals,axis=1,ddof=1),marker=markers[m],color=colors[m],linestyle='-' if m==M[0] else '--',capsize=3,label=names[m])
        ax.set_xticks(range(3),['Mild','Moderate','Severe']); ax.set_ylabel(ylabel)
        if i==0: ax.set_title(label); ax.set_ylim(0,103)
        else: ax.set_ylim(-.005,.70)
axs[0,0].legend(loc='lower left'); save(fig,'prediction_response')
fig,axs=plt.subplots(2,3,figsize=(7.16,4.6),layout='constrained')
for j,(f,label) in enumerate(zip(families,fn)):
    for i,(key,ylabel) in enumerate([('cosine_mean_within_model','Raw cosine'),('normalized_instability_mean','Control-ECDF percentile')]):
        ax=axs[i,j]
        for m in M:
            vals=[[r[key] for r in q if r['model']==m and r['family']==f and r['severity']==s] for s in sevs]
            ax.errorbar(range(3),np.mean(vals,1),yerr=np.std(vals,axis=1,ddof=1),marker=markers[m],color=colors[m],linestyle='-' if m==M[0] else '--',capsize=3,label=names[m])
        ax.set_xticks(range(3),['Mild','Moderate','Severe']); ax.set_ylabel(ylabel)
        if i==0: ax.set_title(label); ax.set_ylim(.25,1.02)
        elif j==2: ax.set_ylim(-.01,.70)
        else: ax.set_ylim(-.0003,.0105); ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axs[0,0].legend(loc='lower left'); save(fig,'representation_response')
fig,axs=plt.subplots(1,2,figsize=(7.16,3.0),layout='constrained')
for ax,metric,title in zip(axs,['average_precision','auroc'],['Incremental AP','Incremental AUROC']):
    rr=[r for r in stats['primary_per_seed'] if r['contrast']=='rq3_nested_detector_delta_'+metric]
    for i,r in enumerate(rr):
        e=r['effect']; lo,hi=r['ci']; ax.errorbar(e,i,xerr=[[e-lo],[hi-e]],fmt=markers[r['model']],color=colors[r['model']],capsize=3)
    ax.set_yticks(range(len(rr)),[names[r['model']]+' / '+str(r['seed']) for r in rr]); ax.invert_yaxis(); ax.axvline(0,color='.3',lw=.8); ax.set_title(title); ax.set_xlabel('Output + representation minus output only'); ax.set_xlim(-.015,.56 if metric=='average_precision' else .26)
save(fig,'detector_deltas')
fig,axs=plt.subplots(1,2,figsize=(7.16,3.9),layout='constrained')
for ax,m in zip(axs,M):
    mat=np.array([[100*np.mean([r['label_consistency'] for r in cl if r['model']==m and r['class_name']==cls and r['family']==f and r['severity']=='severe']) for f in families] for cls in classes])
    im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=100,aspect='auto'); ax.grid(False)
    ax.set_xticks(range(3),['Gain','Mixing','Resolution']); ax.set_yticks(range(10),classes); ax.set_title(names[m])
    for i in range(10):
        for j in range(3): ax.text(j,i,f'{mat[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]>65 else '#202020')
fig.colorbar(im,ax=axs,label='Label consistency (%)',shrink=.85); save(fig,'class_heatmap')
# Re-render training curves directly from recorded event logs.
fig,axs=plt.subplots(1,2,figsize=(7.16,3.0),layout='constrained')
for ax,m in zip(axs,M):
    for seed,style in zip([17,29,43],['-','--',':']):
        events=[json.loads(l) for l in (A/'runs'/f'{m}-seed{seed}'/'events.jsonl').read_text().splitlines()]
        vals=[r for r in events if r.get('event')=='validation' and not r.get('bounded')]
        # Event fields live at top level in the experiment logger.
        ax.plot([r['epoch']+1 for r in vals],[r['macro_f1'] for r in vals],linestyle=style,color=colors[m],label=f'Seed {seed}')
    ax.set_title(names[m]); ax.set_xlabel('Epoch (one-based)'); ax.set_ylabel('Validation macro-F1'); ax.set_ylim(.65,1); ax.legend()
save(fig,'learning_curves')
# Existing supplementary sources are retained for exact lookup, not blindly used
# as headline evidence; main figures above are rebuilt from audited data.
for p in (A/'report_artifacts/final').glob('*.csv'): shutil.copyfile(p,R/'supplementary'/p.name)
for p in (A/'report_artifacts/final').glob('*.png'):
    shutil.copyfile(p,R/'supplementary'/p.name)
for name in ['qualitative_failure_gallery.json','attention_rollout.json']:
    shutil.copyfile(A/'report_artifacts/final'/name,R/'supplementary'/name)
print('ASSETS_BUILT',flush=True)
