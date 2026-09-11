# Environment profiles

- `requirements.txt`: recorded local research software, including Torch 2.13 and torchvision 0.28.
- `requirements-a100-cu124.txt`: recorded A100 CUDA 12.4 environment, Torch 2.6 and torchvision 0.21.
- `requirements-ci.txt`: pinned CPU counterpart for clean CI; includes pytest.

Python 3.12 is the publication target. `pyproject.toml` retains a compatibility range for Torch; a requirements profile supplies exact versions. A compatibility range alone is not an environment lock. Install one profile in a fresh environment and record `python --version` and `python -m pip freeze` with any new execution.

The CPU replay recomputes saved numeric estimates. It does not establish bitwise GPU training reproducibility across hardware. The original H100/A100 difference remains a limitation of training-time comparisons.
