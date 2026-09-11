import os
"""Rebuild requested presentation assets from saved evidence; never run a model.

Only eight named gallery inputs are read from the supplied migration TAR.
No full-archive extraction, package installation, or original-source writes.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import tarfile
import textwrap
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
A = Path(os.environ.get('SAC_EVIDENCE_ROOT', str(R.parent/'all_artifacts')))
F = R / 'figures'
MODELS = ['resnet50', 'vit_small_patch16_224']
NAMES = ['ResNet-50', 'ViT-Small/16']
COLORS = ['#2864A0', '#C27A23']

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(fig, name):
    fig.savefig(F / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(F / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    F.mkdir(exist_ok=True)
    baseline = R / 'audit/revision_baseline.json'
    if not baseline.exists():
        baseline.write_text(json.dumps({
            'report_files': {str(p.relative_to(R)): digest(p) for p in R.rglob('*')
                             if p.is_file() and p.suffix in {'.md', '.tex', '.csv', '.png'}},
            'original_visuals': {p.name: digest(p) for p in (A/'report_artifacts/final').glob('*') if p.is_file()},
            'headlines': read(R/'audit/source_validation.json')['headlines'],
        }, indent=2) + '\n', encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9, 'pdf.fonttype': 42})
    gallery = read(A/'report_artifacts/final/qualitative_failure_gallery.json')
    audit = read(R/'audit/source_audit.json')
    selected = {}
    scatter = {m: [] for m in MODELS}
    source_records = []
    candidates = []
    for run in audit['runs']:
        name = run['run']
        model = name.rsplit('-seed', 1)[0]
        path = A/'evaluations'/name/'final_test/pair_metrics.jsonl'
        index = read(path.parent/'evaluation_index.json')
        class_names = read(A/'runs'/name/'resolved_run.json')['config']['data']['class_names']
        assert digest(path) == index['pair_metrics_sha256'], name
        source_records.append({'run': name, 'pair_sha256': digest(path)})
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['severity'] != 'sham':
                    scatter[model].append((row['confidence_change'], row['normalized_representation_instability']))
                if (row.get('audit_qualified') and row['severity'] == 'mild'
                    and row['family'] in {'spectral_appearance', 'texture'}
                    and max(row['original_probabilities']) >= .99 and row['counterfactual_failure']):
                    candidates.append({**row, 'run': name, 'model': model})
                for meta in gallery['rows']:
                    if meta['run'] == name and meta['pair_id'] == row['pair_id']:
                        assert meta['source_id'] == row['source_id']
                        assert meta['true_class'] == row['class_name'] == class_names[row['label']]
                        assert meta['clean_prediction'] == class_names[int(np.argmax(row['original_probabilities']))]
                        assert meta['transformed_prediction'] == class_names[int(np.argmax(row['transformed_probabilities']))]
                        assert (meta['family'], meta['severity']) == (row['family'], row['severity'])
                        assert abs(meta['clean_confidence'] - max(row['original_probabilities'])) < 1e-12
                        assert abs(meta['transformed_confidence'] - max(row['transformed_probabilities'])) < 1e-12
                        selected[meta['pair_id']] = row
    candidates.sort(key=lambda r: (-max(r['original_probabilities']), r['model'], r['source_id'], r['pair_id']))
    chosen, seen = [], set()
    for row in candidates:
        if row['source_id'] not in seen:
            chosen.append(row); seen.add(row['source_id'])
        if len(chosen) == 4: break
    assert [(r['run'], r['pair_id']) for r in chosen] == [(r['run'], r['pair_id']) for r in gallery['rows']]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), sharex=True, sharey=True, layout='constrained')
    for ax, model, name, color in zip(axes, MODELS, NAMES, COLORS):
        values = np.asarray(scatter[model])
        assert values.shape == (109350, 2) and np.isfinite(values).all()
        assert ((values[:, 1] >= 0) & (values[:, 1] <= 1)).all()
        ax.scatter(values[:, 0], values[:, 1], s=2.5, alpha=.14, color=color, rasterized=True, linewidths=0)
        ax.axvline(0, color='.4', lw=.65, zorder=0)
        ax.set(xlim=(-1, 1), ylim=(-.025, 1.025), title=f'{name} (n = {len(values):,})',
               xlabel='Confidence drop: max p − max q')
        ax.grid(alpha=.13); ax.set_axisbelow(True)
    axes[0].set_ylabel('Representation instability\n(validation-control percentile)')
    save(fig, 'confidence_representation_scatter')

    classes = read(R/'audit/classes.json')
    labels = sorted({r['class_name'] for r in classes})
    families = ['resolution', 'spectral_appearance', 'texture']
    matrix = np.asarray([[np.mean([r['label_consistency'] for r in classes
        if r['model'] == model and r['class_name'] == cls and r['family'] == family and r['severity'] == 'severe'])
        for model in MODELS for family in families] for cls in labels])
    fig, ax = plt.subplots(figsize=(7.16, 4.0), layout='constrained')
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xticks(range(6), ['Resolution', 'RGB gain', 'Amplitude\nmixing'] * 2)
    ax.set_yticks(range(10), labels)
    ax.axvline(2.5, color='white', lw=3)
    ax.text(1, -1.05, NAMES[0], ha='center', weight='bold')
    ax.text(4, -1.05, NAMES[1], ha='center', weight='bold')
    for i in range(10):
        for j in range(6):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if matrix[i,j] < .5 else '#151515')
    fig.colorbar(im, ax=ax, label='Label consistency', fraction=.035, pad=.025)
    save(fig, 'class_heatmap')

    manifest = {}
    with (A/'data/manifest.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['sample_id'] in {m['source_id'] for m in gallery['rows']}:
                manifest[row['sample_id']] = row
    needed = {}
    for meta in gallery['rows']:
        row = selected[meta['pair_id']]
        src = manifest[meta['source_id']]
        needed['data/eurosat/' + row['source_path']] = src['sha256']
        needed['artifacts/' + row['cache_relpath']] = row['cache_sha256']
    payloads = {}
    with tarfile.open(args.archive, 'r|*') as archive:
        for member in archive:
            key = member.name.removeprefix('./')
            if key in needed:
                assert member.isfile(), key
                data = archive.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest() == needed[key], key
                payloads[key] = data
    assert set(payloads) == set(needed), 'Missing gallery source/cache in backup'
    inputs = R/'audit/visual_inputs'
    inputs.mkdir(exist_ok=True)
    visual_rows = []
    for part in range(2):
        fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.2), layout='constrained')
        for k, meta in enumerate(gallery['rows'][part*2:part*2+2]):
            row = selected[meta['pair_id']]
            clean_bytes = payloads['data/eurosat/' + row['source_path']]
            cached_bytes = payloads['artifacts/' + row['cache_relpath']]
            image = np.asarray(Image.open(io.BytesIO(clean_bytes)).convert('RGB')) / 255.
            transformed = np.load(io.BytesIO(cached_bytes), allow_pickle=False)
            assert image.shape == (64,64,3) and transformed.shape == (3,64,64) and transformed.dtype == np.float32
            for ax, img, title in [(axes[k,0], image, f"Clean: {meta['clean_prediction']}\np(max) = {meta['clean_confidence']:.6f}"),
                                   (axes[k,1], transformed.transpose(1,2,0).clip(0,1), f"Mild RGB gain: {meta['transformed_prediction']}\np(max) = {meta['transformed_confidence']:.6f}")]:
                ax.imshow(img, interpolation='nearest')
                ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            label = f"Example {part*2+k+1}: {NAMES[MODELS.index(meta['model'])]}, seed {meta['seed']}\nDataset label: {meta['true_class']}"
            axes[k,0].set_xlabel(label, fontsize=8, labelpad=6)
            # Flat report-local filenames prevent archive path traversal and preserve byte hashes.
            (inputs/(meta['source_id']+'.jpg')).write_bytes(clean_bytes)
            (inputs/(meta['pair_id']+'.npy')).write_bytes(cached_bytes)
            visual_rows.append({**meta, 'source_sha256': needed['data/eurosat/'+row['source_path']],
                                'cache_sha256': row['cache_sha256'], 'source_path': row['source_path']})
        save(fig, f'qualitative_failure_gallery_{part+1}')

    canonical = json.loads((R/'audit/canonical_assets.json').read_text())
    historical = json.loads((R/'audit/visual_revision.json').read_text())
    logos = historical['logos']
    for row in logos:
        rel = 'figures/logos/' + row['file']
        if digest(R/rel) != canonical[rel]:
            raise RuntimeError('Canonical logo changed: ' + rel)
    result = {'sources': source_records, 'scatter_counts': {m:len(v) for m,v in scatter.items()},
              'heatmap': {'classes': labels, 'families': families, 'values': matrix.tolist(), 'scale': [0,1]},
              'gallery_selection_replayed': True, 'gallery_inputs_hash_verified': True, 'gallery_labels_verified': True,
              'gallery_rows': visual_rows, 'logos': logos,
              'archive': str(args.archive), 'neural_inference': False}
    (R/'audit/visual_revision.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('VISUAL_REVISION_BUILT: scatter, heatmap, 4 verified gallery pairs, 4 trimmed logos', flush=True)

if __name__ == '__main__': main()
