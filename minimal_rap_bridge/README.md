# Minimal PufferDrive -> RAP Bridge

This folder provides a minimal viable validation path (Option 1) without changing PufferDrive C internals.

Script:

- `minimal_rap_bridge/render_pufferdrive_to_rap.py`
- `minimal_rap_bridge/sweep_scene_visualize.py`
- `minimal_rap_bridge/make_seed_videos.py`
- `minimal_rap_bridge/compare_state_vs_rap.py`
- `minimal_rap_bridge/compare_native_video_to_rap.py`

What it does:

1. Runs `pufferlib.ocean.drive.Drive`
2. Pulls live state via `get_global_agent_state()` + road edges via `get_road_edge_polylines()`
3. Converts to RAP `ScenarioRenderer.observe(...)` schema
4. Saves rendered camera frames to disk

Current intentional simplifications:

- Uses only road-edge geometry as RAP `BOUNDARY` map features
- Uses constant box height (`--assumed-height`, default `1.6`)
- Uses empty traffic lights (`[]`)
- Keeps map fixed (`resample_frequency=0`)

---

## Run

From repo root:

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap.py \
  --out-dir /tmp/pufferdrive_rap_minimal \
  --frames 30 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 32
```

If you already use the RAP smoke venv, activate it first, then ensure PufferDrive dependencies are available in that env.

---

## Useful Flags

- `--cameras CAM_F0,CAM_L0,CAM_R0`
- `--ego-agent-index 0`
- `--ego-select-mode index|native_visualize`
- `--include-ego-box`
- `--assumed-height 1.6`
- `--map-radius 120`
- `--agent-radius 200` (optional, default: no radius filtering)
- `--render-width 1920 --render-height 1120`
- `--no-rescale-intrinsics` (debug only)
- `--control-mode control_agents`
- `--init-mode create_all_valid`
- `--replay-captured`
- `--replay-source captured|ground_truth`
- `--camera-yaw-mode heading|fixed`
- `--camera-yaw-fixed-rad 0.0`

Note: RAP camera intrinsics are native to `1920x1120`. If you render at another size, the bridge now rescales intrinsics by default to avoid perspective/horizon mismatch.
Note: The bridge now augments `map_features` with lane/road-line polylines parsed from the `.bin` map, in addition to road-edge boundaries.

---

## Output

Images are saved as:

- `/tmp/pufferdrive_rap_minimal/frame_0000_CAM_F0.jpg`
- `/tmp/pufferdrive_rap_minimal/frame_0000_CAM_L0.jpg`
- `/tmp/pufferdrive_rap_minimal/frame_0000_CAM_R0.jpg`
- ...

The script also prints per-frame nonzero pixel counts per camera as a quick sanity signal.
At the end, it prints a QA summary with:

- Per-camera nonzero frame ratio (`nonzero_frames/total_frames`)
- Per-camera nonzero pixel stats (`min/median/mean/max`)
- `map_features_per_frame` stats (`min/median/mean/max`)
- `agent_boxes_per_frame` stats (`min/median/mean/max`)

---

## Multi-Scene Visual Sweep

To test whether sparse/black frames are scene-specific, run multiple seeds and generate one overview image:

```bash
python minimal_rap_bridge/sweep_scene_visualize.py \
  --out-root /tmp/pd_rap_scene_sweep \
  --seeds 1,2,3,4,5,6,7,8 \
  --frames 80 \
  --episode-length 120 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 32 \
  --cameras CAM_F0,CAM_L0,CAM_R0 \
  --render-width 1920 \
  --render-height 1120 \
  --map-radius 100 \
  --ego-agent-index 6 \
  --include-ego-box
```

Outputs:

- `/tmp/pd_rap_scene_sweep/seed_0001/...` (per-seed rendered frames + run log)
- `/tmp/pd_rap_scene_sweep/summary.csv` (per-seed nonzero coverage)
- `/tmp/pd_rap_scene_sweep/overview.jpg` (best frame strip per seed for quick visual check)

---

## Replay-Style Rendering

For dataset-like deterministic playback, capture one rollout first and then render from that captured state sequence:

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap.py \
  --out-dir /tmp/pd_rap_replay_style \
  --frames 80 \
  --episode-length 120 \
  --map-dir /jerry_slow_vol/pufferdrive_womd_val_bins/validation \
  --num-maps 1000 \
  --num-agents 32 \
  --cameras CAM_F0 \
  --ego-agent-index 6 \
  --include-ego-box \
  --replay-captured \
  --replay-source captured \
  --camera-yaw-mode heading
```

Use `--replay-source ground_truth` to replay logged trajectories instead of open-loop rollout. This is recommended for validating geometry alignment, because open-loop stepping can still diverge from logged trajectories.

### Trimmed Baseline Script (known-good defaults)

Use this script as the clean starting point for further closed-loop work. It keeps the known-good settings fixed:
- replay source is always `captured`
- lateral flip is always disabled

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap_baseline.py \
  --out-dir /tmp/pd_rap_baseline \
  --frames 80 \
  --episode-length 120 \
  --seed 1 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 32 \
  --cameras CAM_F0,CAM_L0,CAM_R0 \
  --ego-select-mode native_visualize \
  --include-ego-box \
  --camera-yaw-mode heading
```

For native closed-loop stepping (`forward(net, obs) -> c_step`) with the same trimmed RAP path:

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap_baseline.py \
  --out-dir /tmp/pd_rap_baseline_native_policy \
  --frames 80 \
  --episode-length 120 \
  --seed 1 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 32 \
  --cameras CAM_F0,CAM_L0,CAM_R0 \
  --ego-select-mode native_visualize \
  --include-ego-box \
  --camera-yaw-mode heading \
  --control-source native_policy \
  --policy-path resources/drive/puffer_drive_weights.bin
```

### Native Policy Step-by-Step Test

Run these in order to isolate issues quickly:

1) Verify native-policy APIs exist in binding:
```bash
./.venv-pufferdrive-rap/bin/python -c "from pufferlib.ocean.drive import binding; print(hasattr(binding,'vec_policy_init'), hasattr(binding,'vec_policy_step'), hasattr(binding,'vec_policy_close'))"
```

2) Verify one native policy step works in `Drive`:
```bash
./.venv-pufferdrive-rap/bin/python -c "from pufferlib.ocean.drive.drive import Drive; env=Drive(num_agents=16,num_maps=1,map_dir='resources/drive/binaries',control_mode='control_agents',init_mode='create_all_valid',episode_length=10,resample_frequency=0,report_interval=100); env.reset(seed=1); env.init_native_policy('resources/drive/puffer_drive_weights.bin'); env.step_native_policy(); print('tick', env.tick, 'action0', int(env.actions[0])); env.close()"
```

3) Render 1-frame RAP smoke with native policy:
```bash
./.venv-pufferdrive-rap/bin/python minimal_rap_bridge/render_pufferdrive_to_rap_baseline.py \
  --out-dir /tmp/pd_rap_baseline_native_smoke \
  --frames 1 \
  --episode-length 2 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 16 \
  --cameras CAM_F0 \
  --control-source native_policy
```

4) If step (3) is stable, increase frames and compare against native video workflow.

### State-vs-RAP Baseline Compare (clean script)

Use this for side-by-side debugging panels with the same trimmed conventions.

Captured replay mode:
```bash
python minimal_rap_bridge/compare_state_vs_rap_baseline.py \
  --out-dir /tmp/pd_rap_state_compare_baseline_captured \
  --frames 80 \
  --episode-length 120 \
  --seed 1 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 32 \
  --cameras CAM_F0,CAM_L0,CAM_R0 \
  --panel-camera CAM_F0 \
  --ego-select-mode native_visualize \
  --include-ego-box \
  --control-source captured_replay
```

Native policy closed-loop mode:
```bash
python minimal_rap_bridge/compare_state_vs_rap_baseline.py \
  --out-dir /tmp/pd_rap_state_compare_baseline_native \
  --frames 80 \
  --episode-length 120 \
  --seed 1 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 32 \
  --cameras CAM_F0,CAM_L0,CAM_R0 \
  --panel-camera CAM_F0 \
  --ego-select-mode native_visualize \
  --include-ego-box \
  --control-source native_policy \
  --policy-path resources/drive/puffer_drive_weights.bin
```

### One-Command Pipeline (compare + render + RAP strip video + native-vs-RAP match)

```bash
bash minimal_rap_bridge/run_native_policy_pipeline.sh \
  --native-video /tmp/pd_native_vs_rap/native_agent.mp4 \
  --out-root /tmp/pd_rap_pipeline_native \
  --frames 80 \
  --episode-length 120 \
  --seed 1 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 21 \
  --control-mode control_vehicles \
  --init-mode create_all_valid \
  --cameras CAM_F0,CAM_L0,CAM_R0 \
  --panel-camera CAM_F0 \
  --ego-select-mode native_visualize \
  --include-ego-box \
  --policy-path resources/drive/puffer_drive_weights.bin \
  --strip-cameras CAM_L0,CAM_F0,CAM_R0 \
  --strip-fps 10 \
  --strip-video /tmp/pd_rap_pipeline_native/rap_strip_lfr.mp4 \
  --frame-offset 0 \
  --sample-every 1 \
  --max-samples 80
```

By default this pipeline also:
- logs native-policy actions from a no-render reference loop
- logs native-policy actions during RAP render
- writes an action consistency report (reference vs RAP)
- writes a stitched horizontal RAP video from left/front/right cameras

Outputs under `--out-root` include:
- `native_policy_actions_ref.csv`
- `native_policy_actions_rap.csv`
- `action_log_compare.txt`
- `rap_strip_lfr.mp4`

Use `--skip-action-check` to disable action-log comparison.
Use `--skip-strip-video` to disable stitched RAP strip video generation.

If ego identity still mismatches native, lock ego explicitly:

```bash
bash minimal_rap_bridge/run_native_policy_pipeline.sh \
  --native-video /tmp/pd_native_vs_rap/native_agent.mp4 \
  --out-root /tmp/pd_rap_pipeline_native_ego_locked \
  --frames 80 \
  --episode-length 120 \
  --seed 1 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 21 \
  --control-mode control_vehicles \
  --init-mode create_all_valid \
  --cameras CAM_F0,CAM_L0,CAM_R0 \
  --panel-camera CAM_F0 \
  --ego-select-mode index \
  --ego-agent-index 6 \
  --include-ego-box \
  --policy-path resources/drive/puffer_drive_weights.bin
```

Create short MP4s from rendered frames:

```bash
python minimal_rap_bridge/make_seed_videos.py \
  --sweep-root /tmp/pd_rap_scene_sweep_val_5_clean \
  --seeds 1,2,3 \
  --camera CAM_F0 \
  --fps 4
```

If your player shows green/corrupted frames, force highly compatible output:

```bash
python minimal_rap_bridge/make_seed_videos.py \
  --sweep-root /tmp/pd_rap_scene_sweep_val_5_clean \
  --seeds 1,2,3 \
  --camera CAM_F0 \
  --fps 4 \
  --backend ffmpeg \
  --format mp4
```

And for maximum compatibility across older players:

```bash
python minimal_rap_bridge/make_seed_videos.py \
  --sweep-root /tmp/pd_rap_scene_sweep_val_5_clean \
  --seeds 1,2,3 \
  --camera CAM_F0 \
  --fps 4 \
  --format avi \
  --backend opencv \
  --opencv-codec MJPG
```

---

## State-vs-RAP Validation (No native pixels required)

Build per-frame panels that compare:
- Left: BEV drawn directly from extracted PufferDrive state (road edges + boxes)
- Right: RAP camera render from the same frame/scenario

```bash
python minimal_rap_bridge/compare_state_vs_rap.py \
  --out-dir /tmp/pd_rap_state_compare \
  --frames 80 \
  --episode-length 120 \
  --map-dir /jerry_slow_vol/pufferdrive_womd_val_bins/validation \
  --num-maps 1000 \
  --num-agents 32 \
  --cameras CAM_F0,CAM_L0,CAM_R0 \
  --panel-camera CAM_F0 \
  --ego-agent-index 6 \
  --include-ego-box \
  --agent-radius 200 \
  --replay-captured \
  --replay-source ground_truth \
  --camera-yaw-mode heading
```

Outputs:
- `/tmp/pd_rap_state_compare/panels/frame_0000_compare.jpg`
- `/tmp/pd_rap_state_compare/metrics.csv`

Optional: add `--show-native-window` for manual Drive renderer viewing (interactive/headful only).

---

## Native Visualizer Video vs RAP Frames

If you already exported a native `./visualize` video, compare it to RAP frames:

```bash
python minimal_rap_bridge/compare_native_video_to_rap.py \
  --native-video resources/drive/puffer_drive_weights/video/map_000_topdown.mp4 \
  --rap-dir /tmp/pd_rap_mvp_ego6 \
  --camera CAM_F0 \
  --out-dir /tmp/pd_rap_native_compare \
  --frame-offset 0 \
  --sample-every 5
```

### Known-Good Example (ego359, captured, no lateral flip)

This was a successful apples-to-apples sanity check using:
- native video: `/tmp/pd_native_vs_rap/native_agent.mp4`
- RAP frames: `/tmp/pd_native_vs_rap/rap_frames_ego359_noflip_captured`

```bash
./.venv-pufferdrive-rap/bin/python minimal_rap_bridge/compare_native_video_to_rap.py \
  --native-video /tmp/pd_native_vs_rap/native_agent.mp4 \
  --rap-dir /tmp/pd_native_vs_rap/rap_frames_ego359_noflip_captured \
  --camera CAM_F0 \
  --out-dir /tmp/pd_native_vs_rap/compare_ego359_noflip_captured \
  --frame-offset 0 \
  --sample-every 1 \
  --max-samples 27
```

Outputs:
- `/tmp/pd_native_vs_rap/compare_ego359_noflip_captured/pairs` (per-frame side-by-sides)
- `/tmp/pd_native_vs_rap/compare_ego359_noflip_captured/overview.jpg`
- `/tmp/pd_native_vs_rap/compare_ego359_noflip_captured/metrics.csv`

### MVP Limitations (important)

- Policy mismatch (motion source):
  - Native `./visualize` advances with policy inference each step (`forward(net, obs) -> actions`, then step).
  - This bridge does not run the native policy path; it advances with `env.step(actions)` using neutral or replayed actions/states.
  - Expect traffic differences to grow over time even if frame 0 matches well.
- Camera mismatch:
  - Native `--view agent` is a chase camera.
  - RAP `CAM_F0` is a fixed sensor-style camera with its own extrinsics/intrinsics.
  - Use native-vs-RAP comparisons for geometry/identity sanity, not pixel-accurate parity.

Outputs:
- `/tmp/pd_rap_native_compare/pairs/`
- `/tmp/pd_rap_native_compare/overview.jpg`
- `/tmp/pd_rap_native_compare/metrics.csv`
