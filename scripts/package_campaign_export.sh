#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: package_campaign_export.sh incremental|final [SEED] [OUTPUT_DIR]}
seed=${2:-}
output_dir=${3:-/workspace/exports}
artifacts=/workspace/artifacts
mkdir -p "$output_dir"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
staging=$(mktemp -d)
trap 'rm -rf -- "$staging"' EXIT

copy_path() {
  local source=$1
  test -e "$source" || { echo "Missing export input: $source" >&2; exit 2; }
  local relative=${source#/workspace/}
  mkdir -p "$staging/$(dirname "$relative")"
  cp -a "$source" "$staging/$relative"
}

if [[ "$mode" == incremental ]]; then
  [[ "$seed" =~ ^(17|29|43)$ ]] || { echo "Incremental export requires seed 17, 29, or 43" >&2; exit 2; }
  run="vit_small_patch16_224-seed$seed"
  copy_path "$artifacts/runs/$run/checkpoints/best.pt"
  copy_path "$artifacts/runs/$run/checkpoints/latest.pt"
  for file in summary.json events.jsonl resolved_run.json environment.json model.json wandb_run.json; do copy_path "$artifacts/runs/$run/$file"; done
  copy_path "$artifacts/evaluations/$run/validation"
  copy_path "$artifacts/evaluations/$run/calibration"
  copy_path "$artifacts/detectors/$run.json"
  name="sac-qutab-vit-seed${seed}-${timestamp}"
elif [[ "$mode" == final ]]; then
  for model in resnet50 vit_small_patch16_224; do
    for run_seed in 17 29 43; do
      run="$model-seed$run_seed"
      copy_path "$artifacts/runs/$run"
      copy_path "$artifacts/evaluations/$run"
      copy_path "$artifacts/detectors/$run.json"
    done
  done
  for path in freezes statistics report_artifacts preflight preparation_summary.json run_all_summary.json data counterfactuals review; do
    test -e "$artifacts/$path" && copy_path "$artifacts/$path"
  done
  copy_path /workspace/sac-qutab/configs/core.yaml
  copy_path /workspace/sac-qutab/configs/sac-campaign-v1.yaml
  copy_path /workspace/sac-qutab/src
  copy_path /workspace/sac-qutab/tests
  copy_path /workspace/sac-qutab/docs
  copy_path /workspace/sac-qutab/pyproject.toml
  copy_path /workspace/sac-qutab/requirements.txt
  name="sac-qutab-two-model-final-${timestamp}"
else
  echo "Mode must be incremental or final" >&2
  exit 2
fi

find "$staging" -type d -name wandb -prune -exec rm -rf -- {} +
find "$staging" -type d -name cache -prune -exec rm -rf -- {} +
if find "$staging" -type f \( -name '.env*' -o -iname '*wireguard*' -o -iname '*api*key*' \) | grep -q .; then
  echo "Secret-like file found in export staging" >&2
  exit 2
fi
(cd "$staging" && find . -type f ! -name files.sha256 -print0 | sort -z | xargs -0 sha256sum) > "$staging/files.sha256"
bundle="$output_dir/$name.tar"
tar -cf "$bundle" -C "$staging" .
(cd "$output_dir" && sha256sum "$(basename "$bundle")") > "$bundle.sha256"
split -b 512M -d -a 3 "$bundle" "$bundle.part"
echo "$bundle"
