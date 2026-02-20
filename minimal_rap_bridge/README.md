# Minimal PufferDrive -> RAP Bridge

This directory contains a practical bridge from PufferDrive simulator state to RAP `ScenarioRenderer`, plus validation tools for native-policy parity.

The goal for most users is simple:
- run one command,
- get RAP frames + comparison outputs,
- verify action/state consistency.

## Status (as of February 16, 2026)

Based on `minimal_rap_bridge/docs/PROJECT_PROGRESS_REPORT_2026-02-16.md` and `minimal_rap_bridge/docs/DEBUG_HANDOFF.md`:
- Native-policy action parity is passing in the validated single-env setup.
- Exposed global-state replay parity is also passing in that same setup.
- Visual output is useful for geometry/identity sanity, not pixel-equivalence with native `./visualize`.

## Bridge Docs Folder

Long-form notes and handoff logs are in `minimal_rap_bridge/docs/`:
- `DEBUG_HANDOFF.md`
- `PROJECT_PROGRESS_REPORT_2026-02-16.md`
- `SIMULATION_LOGIC_NOTES.md`

## Who Should Use What

- Use `run_native_policy_pipeline.sh` if you want an end-to-end workflow with outputs ready for review.
- Use baseline Python scripts for targeted debugging or custom experiments.
- Use legacy/debug scripts only when you specifically need older or exploratory tooling.

## Folder Layout

- Root (`minimal_rap_bridge/`): active scripts used by current parity workflow.
- `minimal_rap_bridge/legacy/`: older bridge scripts kept for reference.
- `minimal_rap_bridge/debug/`: helper utilities for sweeps/auxiliary video generation.
- `minimal_rap_bridge/docs/`: reports, handoffs, and logic/interface notes.

## Prerequisites

Run from repo root.

1. Build a minimal working env:

```bash
bash envs/pufferdrive_rap_minimal/setup_env.sh
```

2. Activate it:

```bash
source .venv-pufferdrive-rap/bin/activate
```

3. Optional sanity check:

```bash
python envs/pufferdrive_rap_minimal/validate_env.py --check-drive --map-dir resources/drive/binaries
```

Alternative one-command bootstrap from this folder:

```bash
bash minimal_rap_bridge/setup_cluster_env.sh
```

## Recommended Quickstart (One Command)

If you already have a native agent video (`.mp4`), run:

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

Why `--num-agents 21`?
- This is the validated single-env setting for this map/config.
- The pipeline explicitly refuses `num_envs > 1` because merged multi-env output breaks parity checks.

## RAP-Only Quickstart (No Native Video Needed)

If you only need RAP renders (no native video matching), run:

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap_baseline.py \
  --out-dir /tmp/pd_rap_baseline \
  --frames 80 \
  --episode-length 120 \
  --seed 1 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 21 \
  --cameras CAM_F0,CAM_L0,CAM_R0 \
  --ego-select-mode random_seeded \
  --ego-random-seed 1 \
  --include-ego-box \
  --control-source native_policy \
  --policy-path resources/drive/puffer_drive_weights.bin
```

This writes `frame_XXXX_<CAM>.jpg` files and prints per-frame nonzero pixel counts plus a QA summary at the end.

## Renderer Throughput Profiling

Use `profile_rap_renderer_speed.py` to benchmark RAP rendering throughput across fixed map ids.

What it does:
- Stages deterministic one-map directories under `/tmp/rap_profile_maps/map_XXX/`.
- Runs `render_pufferdrive_to_rap.py` for each requested map.
- Renders all 8 camera views:
  - `CAM_F0,CAM_L0,CAM_L1,CAM_L2,CAM_R0,CAM_R1,CAM_R2,CAM_B0`
- Writes per-frame timing CSV and per-scene summary JSON.
- Writes combined aggregate summaries (`profile_summary.csv` / `.json`).

Timing definition:
- Per-frame timing is `renderer.observe(...) + JPEG write`.
- Scene total uses render+write accumulation (`--profile-render-write-only`).

Stability note:
- In some builds, replay modes can crash native code.
- The profiling script therefore runs with replay off and neutral-action stepping for stable throughput measurement.

Smoke test (1 map, 5 frames):

```bash
python minimal_rap_bridge/profile_rap_renderer_speed.py \
  --map-root resources/drive/binaries/validation \
  --map-ids 0 \
  --frames 5 \
  --episode-length 10 \
  --out-root /tmp/rap_renderer_profile_smoke \
  --clean-stage-root
```

Full requested run (5 maps x 91 frames):

```bash
python minimal_rap_bridge/profile_rap_renderer_speed.py \
  --map-root resources/drive/binaries/validation \
  --map-ids 0,1,2,3,4 \
  --frames 91 \
  --episode-length 120 \
  --out-root /tmp/rap_renderer_profile \
  --clean-stage-root
```

Example run (10 manually picked random-ish maps, excluding the first 10):

```bash
python minimal_rap_bridge/profile_rap_renderer_speed.py \
  --map-root resources/drive/binaries/validation \
  --map-ids 12,19,27,34,41,48,56,63,70,74 \
  --frames 91 \
  --episode-length 120 \
  --out-root /tmp/rap_renderer_profile_10random \
  --clean-stage-root
```

Outputs:
- Per-scene:
  - `/tmp/rap_renderer_profile/map_XXX/frames/frame_XXXX_<CAM>.jpg`
  - `/tmp/rap_renderer_profile/map_XXX/timing.csv`
  - `/tmp/rap_renderer_profile/map_XXX/scene_summary.json`
  - `/tmp/rap_renderer_profile/map_XXX/run.log`
- Aggregate:
  - `/tmp/rap_renderer_profile/profile_summary.csv`
  - `/tmp/rap_renderer_profile/profile_summary.json`

## What This Pipeline Runs

`run_native_policy_pipeline.sh` executes, in order:
1. Native-policy reference action log (`log_native_policy_actions_reference.py`)
2. BEV-vs-RAP compare panels (`compare_state_vs_rap_baseline.py`)
3. RAP frame rendering (`render_pufferdrive_to_rap_baseline.py`)
4. Optional stitched RAP strip video (`make_rap_strip_video.py`)
5. Native-vs-RAP side-by-side matching (`compare_native_video_to_rap.py`)
6. Action log parity report (`compare_action_logs.py`)

## Pipeline Output Layout

Under `--out-root`:
- `state_compare/` (BEV+RAP panels + metrics)
- `rap_frames/` (rendered `frame_XXXX_<CAM>.jpg`)
- `native_vs_rap/` (side-by-side pairs + overview + metrics)
- `rap_strip_lfr.mp4` (unless `--skip-strip-video`)
- `native_policy_actions_ref.csv`
- `native_policy_actions_rap.csv`
- `action_log_compare.txt`

## Common Pipeline Flags

- `--native-ego-log <path>`: override RAP ego selection from native visualize log metadata.
- `--ego-select-mode random_seeded --ego-random-seed <N>`: reproducible random ego.
- `--ego-select-mode index --ego-agent-index <N>`: hard-lock ego slot.
- `--match-native-length` (default on): clamp RAP horizon to native video frame count.
- `--frame-offset <N>`: align native/RAP timing if one starts later.
- `--sample-every <N> --max-samples <N>`: control native-vs-RAP pair sampling density.
- `--skip-strip-video`: skip stitched RAP video generation.
- `--skip-action-check`: skip reference-vs-RAP action log compare.

## Interpreting Parity Results

In `action_log_compare.txt`, a pass is:
- `row_count_match=True`
- `mismatch_rows=0`

For deterministic action->state replay parity, run:

```bash
./.venv-pufferdrive-rap/bin/python minimal_rap_bridge/check_policy_action_replay_consistency.py \
  --frames 80 \
  --episode-length 120 \
  --seed 1 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 21 \
  --control-mode control_vehicles \
  --init-mode create_all_valid \
  --ego-select-mode native_visualize \
  --policy-path resources/drive/puffer_drive_weights.bin \
  --out-report /tmp/pd_action_replay_consistency_report.txt \
  --out-csv /tmp/pd_action_replay_consistency.csv
```

A pass report includes:
- `all_steps_match=True`
- `num_mismatch_steps=0`

## If You Need a Native Video First

If you do not already have native `./visualize` output, generate one and keep the log for ego metadata:

```bash
NO_TRAIN=1 python setup.py build_ext --inplace --force

SEED=1
OUT_NATIVE=/tmp/pd_native_vs_rap_seed${SEED}
mkdir -p "${OUT_NATIVE}"

ASAN_OPTIONS=detect_leaks=0 LSAN_OPTIONS=detect_leaks=0 \
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s "-screen 0 1920x1120x24 -ac" \
  ./visualize \
  --map-name resources/drive/binaries/map_000.bin \
  --policy-name resources/drive/puffer_drive_weights.bin \
  --view agent \
  --ego-random-seed "${SEED}" \
  --output-topdown "${OUT_NATIVE}/native_topdown.mp4" \
  --output-agent "${OUT_NATIVE}/native_agent.mp4" \
  | tee "${OUT_NATIVE}/native_agent.log"
```

Then pipeline with ego override from native log:

```bash
bash minimal_rap_bridge/run_native_policy_pipeline.sh \
  --native-video "${OUT_NATIVE}/native_agent.mp4" \
  --native-ego-log "${OUT_NATIVE}/native_agent.log" \
  --out-root /tmp/pd_rap_pipeline_seed${SEED} \
  --frames 80 \
  --episode-length 120 \
  --seed "${SEED}" \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 21 \
  --match-native-length
```

## Script Map

| Script | Purpose | Typical use |
|---|---|---|
| `run_native_policy_pipeline.sh` | Full native-policy parity workflow | Default entrypoint for users |
| `render_pufferdrive_to_rap_baseline.py` | Render RAP frames with baseline defaults | RAP-only rendering / action logging |
| `compare_state_vs_rap_baseline.py` | BEV-from-state vs RAP camera panels | Geometry sanity without native pixels |
| `compare_native_video_to_rap.py` | Native video vs RAP frame matching | Side-by-side visual checks |
| `make_rap_strip_video.py` | Stitched multi-camera RAP video | Quick review videos |
| `log_native_policy_actions_reference.py` | Reference action logging | Pipeline internals / debugging |
| `compare_action_logs.py` | Action parity report | Verify loop consistency |
| `check_policy_action_replay_consistency.py` | Action replay -> state parity checker | Deterministic parity audit |
| `render_pufferdrive_to_rap.py` | Shared core + legacy full-feature bridge script | Baseline dependency / advanced debugging |
| `legacy/compare_state_vs_rap.py` | Legacy full-feature compare script | Advanced debugging only |
| `debug/sweep_scene_visualize.py` | Multi-seed sweep + overview image | Coverage/sparsity analysis |
| `debug/make_seed_videos.py` | Video export from sweep folders | Presentation / QA playback |

## Common Pitfalls

- Multi-env parity drift:
  - Symptom: unrealistic merged traffic and parity breaks.
  - Fix: keep `num_envs=1` (for current baseline, `--num-agents 21`).

- Ego mismatch vs native run:
  - Use `--native-ego-log` when available.
  - Or lock ego explicitly with `--ego-select-mode index --ego-agent-index N`.

- Native vs RAP visuals do not look identical:
  - Expected. Native uses chase-camera rendering; RAP uses fixed sensor camera models.
  - Compare scene geometry, identity continuity, and gross motion trends.

- Missing map binaries:
  - Ensure `--map-dir` points to a folder containing files like `map_000.bin`.

## Additional References

- `minimal_rap_bridge/docs/DEBUG_HANDOFF.md`: chronological debug findings and root causes.
- `minimal_rap_bridge/docs/PROJECT_PROGRESS_REPORT_2026-02-16.md`: parity milestones and roadmap.
- `minimal_rap_bridge/docs/SIMULATION_LOGIC_NOTES.md`: termination/collision/multi-env/trajectory logic notes.
