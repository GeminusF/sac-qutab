import os
"""Compact presentation copies of frozen response curves; no inference."""
from pathlib import Path
import json
import hashlib
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
rows=json.loads((R/'audit/conditions.json').read_text())
models=['resnet50','vit_small_patch16_224']
names=['ResNet-50','ViT-Small/16']
families=['spectral_appearance','texture','resolution']
severities=['mild','moderate','severe']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8.5,
    'ytick.labelsize':8.5,'legend.fontsize':8.5,'pdf.fonttype':42})
for name,keys,labels in [
    ('prediction_response',['transformed_accuracy','js_mean'],['Accuracy (%)','JS divergence (nats)']),
    ('representation_response',['cosine_mean_within_model','normalized_instability_mean'],['Raw cosine','Control-ECDF percentile'])]:
    fig,axes=plt.subplots(2,3,figsize=(7.0,2.7),layout='constrained')
    for j,family in enumerate(families):
        for i,key in enumerate(keys):
            ax=axes[i,j]
            for model,title,color,marker,style in zip(models,names,['#2864A0','#C27A23'],['o','s'],['-','--']):
                values=np.array([[r[key] for r in rows if r['model']==model and r['family']==family and r['severity']==severity] for severity in severities])
                assert values.shape==(3,3)
                if key=='transformed_accuracy':values*=100
                ax.errorbar(range(3),values.mean(1),yerr=values.std(1,ddof=1),color=color,marker=marker,linestyle=style,capsize=2,markersize=3,label=title)
            ax.set_xticks(range(3),['Mild','Moderate','Severe'])
            ax.grid(alpha=.2)
            if j==0:ax.set_ylabel(labels[i])
            if i==0:ax.set_title(['RGB gain','Amplitude mixing','Resolution'][j])
            if key=='transformed_accuracy':ax.set_ylim(0,103)
            elif key=='js_mean':ax.set_ylim(-.005,.70)
            elif key=='cosine_mean_within_model':ax.set_ylim(.25,1.02)
            elif j==2:ax.set_ylim(-.01,.70)
            else:ax.set_ylim(-.0003,.0105);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[0,0].legend(loc='lower left',framealpha=.8,handlelength=1.2,borderpad=.2,labelspacing=.2)
    for extension in ['pdf','png']:
        fig.savefig(R/'figures'/f'{name}_compact.{extension}',dpi=180,bbox_inches='tight',pad_inches=.02)
    plt.close(fig)

def save(fig,name):
    for extension in ['pdf','png']:
        fig.savefig(R/'figures'/f'{name}_compact.{extension}',dpi=180,bbox_inches='tight',pad_inches=.02)
    plt.close(fig)

# All non-sham observations, exactly as in the archived scatter contract.
fig,axes=plt.subplots(1,2,figsize=(7,2.5),sharex=True,sharey=True,layout='constrained')
for ax,model,name,color in zip(axes,models,names,['#2864A0','#C27A23']):
    points=[]
    for seed in [17,29,43]:
        path=Path(os.environ.get('SAC_EVIDENCE_ROOT', str(R.parent/'all_artifacts')))/'evaluations'/f'{model}-seed{seed}'/'final_test/pair_metrics.jsonl'
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row['severity']!='sham':points.append((row['confidence_change'],row['normalized_representation_instability']))
    values=np.asarray(points)
    assert values.shape==(109350,2)
    ax.scatter(values[:,0],values[:,1],s=2.5,alpha=.14,color=color,rasterized=True,linewidths=0)
    ax.axvline(0,color='.4',lw=.65,zorder=0)
    ax.set(xlim=(-1,1),ylim=(-.025,1.025),title=f'{name} (n = {len(values):,})',xlabel='Confidence drop: max p - max q')
    ax.grid(alpha=.13);ax.set_axisbelow(True)
axes[0].set_ylabel('Representation instability\n(validation-control percentile)')
save(fig,'confidence_representation_scatter')

classes=json.loads((R/'audit/classes.json').read_text())
labels=sorted({r['class_name'] for r in classes})
matrix=np.asarray([[np.mean([r['label_consistency'] for r in classes if r['model']==model and r['class_name']==cls and r['family']==family and r['severity']=='severe']) for model in models for family in ['resolution','spectral_appearance','texture']] for cls in labels])
fig,ax=plt.subplots(figsize=(7,3.2),layout='constrained')
im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
ax.set_xticks(range(6),['Resolution','RGB gain','Amplitude\nmixing']*2)
ax.set_yticks(range(10),labels);ax.axvline(2.5,color='white',lw=3)
for x,name in [(1,names[0]),(4,names[1])]:ax.text(x,-1.05,name,ha='center',weight='bold')
for i in range(10):
    for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8.5,color='white' if matrix[i,j]<.5 else '#151515')
fig.colorbar(im,ax=ax,label='Label consistency',fraction=.035,pad=.025)
save(fig,'class_heatmap')

fig,axes=plt.subplots(1,2,figsize=(7,2.0),layout='constrained')
for ax,model,name,color in zip(axes,models,names,['#2864A0','#C27A23']):
    for seed,style in zip([17,29,43],['-','--',':']):
        path=Path(os.environ.get('SAC_EVIDENCE_ROOT', str(R.parent/'all_artifacts')))/'runs'/f'{model}-seed{seed}'/'events.jsonl'
        events=[json.loads(line) for line in path.read_text().splitlines()]
        vals=[r for r in events if r.get('event')=='validation' and not r.get('bounded')]
        ax.plot([r['epoch']+1 for r in vals],[r['macro_f1'] for r in vals],linestyle=style,color=color,label=f'Seed {seed}')
    ax.set(title=name,xlabel='Epoch (one-based)',ylabel='Validation macro-F1',ylim=(.65,1))
    ax.legend(loc='lower right',ncol=3,handlelength=1,borderpad=.2,columnspacing=.6)
save(fig,'learning_curves')

gallery=json.loads((R/'audit/visual_revision.json').read_text())
metadata=json.loads((R/'supplementary/qualitative_failure_gallery.json').read_text())['rows']
for part in range(2):
    fig,axes=plt.subplots(1,4,figsize=(7,2.65),layout='constrained')
    for k,meta in enumerate(metadata[part*2:part*2+2]):
        clean=R/'audit/visual_inputs'/(meta['source_id']+'.jpg')
        cache=R/'audit/visual_inputs'/(meta['pair_id']+'.npy')
        record=next(r for r in gallery['gallery_rows'] if r['pair_id']==meta['pair_id'])
        assert hashlib.sha256(clean.read_bytes()).hexdigest()==record['source_sha256']
        assert hashlib.sha256(cache.read_bytes()).hexdigest()==record['cache_sha256']
        images=[np.asarray(Image.open(clean).convert('RGB'))/255,np.load(cache,allow_pickle=False).transpose(1,2,0).clip(0,1)]
        for j,im in enumerate(images):
            ax=axes[k*2+j]
            prediction=meta['clean_prediction' if j==0 else 'transformed_prediction']
            prediction=prediction.replace('HerbaceousVegetation','Herbaceous\nVegetation').replace('PermanentCrop','Permanent\nCrop')
            confidence=meta['clean_confidence' if j==0 else 'transformed_confidence']
            ax.imshow(im,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(('Clean' if j==0 else 'Mild RGB gain')+'\n'+prediction+f'\np(max) = {confidence:.6f}',fontsize=8.5)
            if j==0:ax.set_xlabel(f'Example {part*2+k+1}\n'+names[models.index(meta['model'])]+f", seed {meta['seed']}",fontsize=8.5)
    save(fig,f'qualitative_failure_gallery_{part+1}')

# Re-layout the already verified maps; no model loading or forward calls.
meta=metadata[1]
raw=[np.asarray(Image.open(R/'audit/visual_inputs'/(meta['source_id']+'.jpg')).convert('RGB'))/255,
     np.load(R/'audit/visual_inputs'/(meta['pair_id']+'.npy'),allow_pickle=False).transpose(1,2,0).clip(0,1)]
heat=np.load(R/'audit/attention_rollout_raw.npz',allow_pickle=False)['heat']
fig,axes=plt.subplots(1,4,figsize=(7,2.2),layout='constrained')
for i,title in enumerate(['Clean','Mild RGB gain']):
    for j in range(2):
        ax=axes[i*2+j]
        ax.imshow(raw[i],extent=(0,224,224,0),interpolation='nearest')
        if j:ax.imshow(heat[i],extent=(0,224,224,0),cmap='jet',alpha=.45,vmin=0,vmax=1)
        ax.set_title(title+('\nrollout (per-image scale)' if j else '\ninput'),fontsize=8.5)
        ax.set_xticks([]);ax.set_yticks([])
save(fig,'attention_rollout')
