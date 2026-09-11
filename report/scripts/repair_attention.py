import os
"""Optional, bounded post-hoc repair: one fixed pair, one frozen checkpoint.

Requires an existing compatible runtime; never installs/downloads dependencies.
--check-only performs no model inference. Outputs stay under report/.
The produced plot must be visually reviewed before manuscript inclusion.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys

R = Path(__file__).resolve().parents[1]
A = Path(os.environ.get('SAC_EVIDENCE_ROOT', str(R.parent/'all_artifacts')))
RUN = 'vit_small_patch16_224-seed17'
PAIR = '70b6fc0226c23da482d541e893906a2d05f67626d9ece12eb556090872889420'

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    required = ['torch', 'torchvision', 'timm', 'numpy', 'matplotlib', 'PIL']
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    status = {'status': 'BLOCKED' if missing else 'RUNTIME_DISCOVERED', 'missing_modules': missing,
              'run': RUN, 'pair_id': PAIR, 'inference_performed': False,
              'packages_installed': False, 'figure_visually_verified': False}
    if not missing and importlib.metadata.version('timm') != '1.0.29':
        status.update(status='BLOCKED', reason='Requires recorded timm==1.0.29; no automatic upgrade/downgrade')
    target = R/'audit/attention_repair_status.json'
    if status['status'] == 'BLOCKED' or args.check_only:
        target.write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(status)); return 2 if status['status'] == 'BLOCKED' else 0

    import numpy as np
    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(R.parent/'src'))
    from sac_qutab.config import ModelPreset
    from sac_qutab.models import build_model
    from sac_qutab.data import preprocess_image
    from sac_qutab.interpretability import attention_rollout

    protocol_path = A/'freezes/experiment.json'
    protocol = read(protocol_path)
    assert digest(protocol_path) == read(R/'audit/source_audit.json')['protocol_sha256']
    checkpoint_path = A/'runs'/RUN/'checkpoints/best.pt'
    resolved_path = A/'runs'/RUN/'resolved_run.json'
    assert digest(checkpoint_path) == protocol['checkpoint_sha256'][RUN]
    resolved = read(resolved_path)
    # Frozen config_digest hashes canonical JSON, not its indented disk bytes.
    canonical = json.dumps(resolved, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == protocol['resolved_run_sha256'][RUN]
    record = next(r for r in read(R/'audit/visual_revision.json')['gallery_rows'] if r['pair_id'] == PAIR)
    clean_path = R/'audit/visual_inputs'/(record['source_id']+'.jpg')
    cache_path = R/'audit/visual_inputs'/(PAIR+'.npy')
    assert digest(clean_path) == record['source_sha256']
    assert digest(cache_path) == record['cache_sha256']
    clean = torch.from_numpy(np.asarray(Image.open(clean_path).convert('RGB')).copy()).permute(2,0,1).float()/255
    cached = np.load(cache_path, allow_pickle=False)
    assert cached.dtype == np.float32 and cached.shape == (3,64,64)
    transformed = torch.from_numpy(cached.copy())
    batch = torch.stack([preprocess_image(clean), preprocess_image(transformed)])
    torch.set_num_threads(2)
    model = build_model(ModelPreset(**resolved['model']), 10).cpu().eval()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['schema_version'] == 'sac-checkpoint-v2'
    assert checkpoint['resolved_run_sha256'] == protocol['resolved_run_sha256'][RUN]
    model.load_state_dict(checkpoint['model'], strict=True)
    captured = []
    handle = model.register_forward_hook(lambda _m, _i, o: captured.append(o.logits.detach()))
    try:
        heat = attention_rollout(model, batch)
    finally:
        handle.remove()
    assert len(captured) == 1 and heat.shape == (2,224,224) and np.isfinite(heat).all()
    probabilities = captured[0].softmax(-1).numpy()
    classes = resolved['config']['data']['class_names']
    predicted = [classes[i] for i in probabilities.argmax(1)]
    assert predicted == [record['clean_prediction'], record['transformed_prediction']], 'Prediction mismatch'
    # CPU float32 replay is a post-hoc visualization, not bitwise GPU reproduction.
    error = float(np.max(np.abs(probabilities.max(1) - [record['clean_confidence'], record['transformed_confidence']])))
    assert error <= .005, 'Confidence mismatch exceeds predeclared 0.005 absolute tolerance'
    raw = [clean.permute(1,2,0).numpy(), transformed.permute(1,2,0).clamp(0,1).numpy()]
    fig, axes = plt.subplots(2,2,figsize=(6.5,6.1),layout='constrained')
    extent = (0,224,224,0)
    for i,title in enumerate(['Clean', 'Mild RGB gain']):
        for ax in axes[i]:
            ax.imshow(raw[i], extent=extent, interpolation='nearest')
            ax.set_xlim(0,224); ax.set_ylim(224,0); ax.set_axis_off()
        axes[i,0].set_title(title+' input')
        axes[i,1].imshow(heat[i], extent=extent, cmap='jet', alpha=.45, vmin=0, vmax=1)
        axes[i,1].set_title(title+' rollout (per-image scale)')
    for ext in ['png','pdf']:
        fig.savefig(R/'figures'/('attention_rollout_repaired.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    np.savez(R/'audit/attention_rollout_raw.npz', heat=heat, probabilities=probabilities)
    status.update(status='GENERATED_AWAITING_VISUAL_REVIEW', inference_performed=True,
                  forward_calls=1, input_images=2, checkpoint_sha256=digest(checkpoint_path),
                  source_sha256=digest(clean_path), cache_sha256=digest(cache_path),
                  source_code_sha256=digest(R.parent/'src/sac_qutab/interpretability.py'),
                  confidence_max_absolute_difference=error, confidence_tolerance=.005,
                  predictions=predicted, device='cpu', dtype='float32',
                  versions={m:importlib.metadata.version(m) for m in ['torch','torchvision','timm','numpy']})
    target.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(status)); return 0

if __name__ == '__main__': raise SystemExit(main())
