# PufferDrive

PufferDrive is a fast driving simulator for RL training and evaluation.  
This repo also includes a practical RAP bridge workflow for rendering and parity checks.

## What This Repo Is For

- Train and evaluate driving policies in PufferDrive.
- Run native visualizer rollouts.
- Convert PufferDrive state into RAP camera renders and compare native vs RAP behavior.

If your focus is the RAP bridge pipeline, start here:
- `minimal_rap_bridge/README.md`

## Bridge Intent and Approach

The bridge workflow (`minimal_rap_bridge/`) is designed for reproducible native-policy parity checks:

- Use native policy stepping (`step_native_policy`) in PufferDrive.
- Render RAP camera outputs from simulator state.
- Compare action/state parity and side-by-side visuals against native output.
- Enforce single-env parity configs to avoid merged multi-scene artifacts.

Recommended entrypoint:

```bash
bash minimal_rap_bridge/run_native_policy_pipeline.sh \
  --native-video /tmp/pd_native_vs_rap/native_agent.mp4 \
  --out-root /tmp/pd_rap_pipeline_native_singleenv \
  --frames 27 \
  --episode-length 120 \
  --seed 1 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 21
```

Detailed bridge script map and per-file roles are documented in:
- `minimal_rap_bridge/README.md`

Long-form bridge notes are in:
- `minimal_rap_bridge/docs/README.md`

## Setup

### Option A: Minimal Bridge Environment (recommended for RAP pipeline)

```bash
bash envs/pufferdrive_rap_minimal/setup_env.sh
source .venv-pufferdrive-rap/bin/activate
python envs/pufferdrive_rap_minimal/validate_env.py --check-drive --map-dir resources/drive/binaries
```

### Option B: General Development Install

```bash
git clone https://github.com/Emerge-Lab/PufferDrive.git
cd PufferDrive
uv venv
source .venv/bin/activate
uv pip install -e .
python setup.py build_ext --inplace --force
```

## Core Simulator Quickstart

Train:

```bash
puffer train puffer_drive
```

Eval WOSAC realism:

```bash
puffer eval puffer_drive --eval.wosac-realism-eval True
```

## Native Visualizer (Headless Example)

Build:

```bash
bash scripts/build_ocean.sh visualize local
```

Run on headless machine:

```bash
xvfb-run -s "-screen 0 1280x720x24" ./visualize
```

## Repository Map

- `pufferlib/ocean/drive/`: simulator core and C/Python bindings
- `minimal_rap_bridge/`: RAP bridge scripts and parity tooling
- `envs/pufferdrive_rap_minimal/`: minimal environment bootstrap for bridge workflows
- `docs/`: project docs site

## Additional Docs

- Project docs: https://emerge-lab.github.io/PufferDrive
- Bridge detailed readme: `minimal_rap_bridge/README.md`
- Bridge notes index: `minimal_rap_bridge/docs/README.md`

## Citation

```bibtex
@software{pufferdrive2025github,
  author = {Daphne Cornelisse* and Spencer Cheng* and Pragnay Mandavilli and Julian Hunt and Kevin Joseph and Waël Doulazmi and Valentin Charraut and Aditya Gupta and Joseph Suarez and Eugene Vinitsky},
  title = {{PufferDrive}: A Fast and Friendly Driving Simulator for Training and Evaluating {RL} Agents},
  url = {https://github.com/Emerge-Lab/PufferDrive},
  version = {2.0.0},
  year = {2025},
}
```
