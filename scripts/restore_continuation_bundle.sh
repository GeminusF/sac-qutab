#!/usr/bin/env bash
set -euo pipefail

parts_dir=${1:?usage: restore_continuation_bundle.sh PARTS_DIR [RUNTIME_ROOT]}
runtime_root=${2:-/workspace}
expected_file="$parts_dir/sac-qutab-continuation.tar.sha256"
bundle="$parts_dir/sac-qutab-continuation.tar"

test -f "$expected_file"
mapfile -t parts < <(find "$parts_dir" -maxdepth 1 -type f -name 'sac-qutab-continuation.tar.part*' | sort)
test "${#parts[@]}" -gt 0
if [[ -e "$bundle" ]]; then
  echo "Refusing to overwrite existing bundle: $bundle" >&2
  exit 2
fi
for part in "${parts[@]}"; do cat "$part" >> "$bundle"; done
(cd "$parts_dir" && sha256sum -c "$(basename "$expected_file")")

if [[ "$runtime_root" == "/workspace" ]]; then
  mkdir -p /workspace
  test -w /workspace || { echo "/workspace must be writable, or pass a writable physical root as argument 2" >&2; exit 2; }
  physical_root=/workspace
  relocation_mechanism=direct
else
  mkdir -p "$runtime_root"
  test -w "$runtime_root" || { echo "Physical runtime root is not writable: $runtime_root" >&2; exit 2; }
  physical_root=$(cd "$runtime_root" && pwd -P)
  relocation_mechanism=application-path-map
fi

tar -tf "$bundle" | while IFS= read -r entry; do
  case "/$entry/" in *"/../"*) echo "Unsafe archive entry: $entry" >&2; exit 2;; esac
  [[ "$entry" != /* ]] || { echo "Absolute archive entry: $entry" >&2; exit 2; }
done
tar -xf "$bundle" -C "$physical_root"

python "$physical_root/sac-qutab/scripts/verify_migration_manifest.py" "$physical_root/migration-manifest.json" "$physical_root"
evidence="$physical_root/artifacts/preflight/a100_mig"
mkdir -p "$evidence"
df -h > "$evidence/df-h.txt"
nvidia-smi -L > "$evidence/nvidia-smi-L.txt"
nvidia-smi -q > "$evidence/nvidia-smi-q.txt"
python --version > "$evidence/python-version.txt" 2>&1
python -m pip freeze > "$evidence/pip-freeze.txt"
python - <<'PY' > "$evidence/torch-cuda.json"
import json, torch
print(json.dumps({
    "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "device_properties": str(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else None,
}, indent=2))
PY
cat > "$evidence/path-relocation.json" <<JSON
{"schema_version":"sac-path-relocation-v1","logical_root":"/workspace","physical_root":"$physical_root","mechanism":"$relocation_mechanism","checkpoint_metadata_rewritten":false}
JSON
cat > "$physical_root/sac-qutab-runtime.env" <<ENV
export SAC_QUTAB_LOGICAL_ROOT=/workspace
export SAC_QUTAB_PHYSICAL_ROOT='$physical_root'
export PYTHONPATH='$physical_root/sac-qutab/src'
ENV
echo "Restore and per-file verification complete under $physical_root"
