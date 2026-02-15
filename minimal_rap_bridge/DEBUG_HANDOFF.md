# PufferDrive -> RAP Bridge Debug Handoff

Date: 2026-02-14
Repo: `/root/RL_Finetune_Project`
Primary scripts:
- `minimal_rap_bridge/render_pufferdrive_to_rap.py`
- `minimal_rap_bridge/compare_state_vs_rap.py`
- `minimal_rap_bridge/compare_native_video_to_rap.py`

## Goal
Build an MVP bridge from PufferDrive state to RAP `ScenarioRenderer` for RL dataset curation, with stable camera behavior and plausible map/agent alignment.

## High-Level Status
- RAP renderer import/smoke works.
- Environment validation for PufferDrive + RAP dependencies works.
- Bridge renders frames and panels, but there have been repeated issues with:
- rotating/spinning vehicle appearance over time
- map edge/lane visibility depending on coordinate treatment
- distribution mismatch versus native `visualize` output

## Core Findings
- Rotation/drift issue was strongly linked to replay semantics and identity handling.
- Using captured rollout with zero-action stepping can diverge from logged trajectories.
- Ground-truth replay reduces many artifacts, but user wants captured-style behavior for certain checks.
- Map features and agent boxes must be in the same coordinate frame.
- Native visualizer displays richer road geometry than road-edge-only extraction.

## Major Changes Applied (chronological intent)
- Added replay options:
- `--replay-captured`
- `--replay-source captured|ground_truth`
- Added ego identity stability by id across frames:
- `resolve_ego_index_by_id(...)`
- Removed camera smoothing/clamp path for MVP simplicity:
- removed `smoothed` mode and yaw-rate clamp path
- Added lane polyline augmentation from map binary:
- parse ROAD_LANE/ROAD_LINE from `.bin`
- inject as RAP map features with `type` containing `LANE`
- Added native-like ego selection option:
- `--ego-select-mode index|native_visualize`
- `native_visualize` emulates first `rand()%active_agent_count` choice used by C visualizer startup logic
- Added static map safety clamping for renderer stability to avoid `cv2.clipLine` type errors.
- Converted static-style map features to ego-relative per frame so map and boxes share frame.
- Added heading correction for captured replay:
- for `captured` source, overwrite per-frame heading by `(agent_id, t)` from GT trajectory while keeping captured positions

## Current Behavior You Should Expect
- More cars visible than earlier sparse runs.
- Lane-like lines present (from map binary parsing) plus boundaries.
- Ego can be changed by index or native-style chooser.
- Some runs may still visually mismatch native due to policy rollout dynamics and renderer style differences.

## Important Known Risks
- C/C++ allocator instability (`malloc corrupted top size` / `sysmalloc assertion`) occurred intermittently during repeated short test invocations in this chat session. This may be environment-state dependent.
- Captured replay can still differ from native trajectory evolution if policy/action paths differ.
- Native visualizer and RAP are not pixel-equivalent; compare geometry trends, not appearance.

## Repro Commands (recommended baseline)

### 1) BEV + RAP compare panel run
```bash
python minimal_rap_bridge/compare_state_vs_rap.py \
  --out-dir /tmp/pd_rap_compare_native_aligned \
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
  --replay-captured \
  --replay-source captured \
  --camera-yaw-mode heading
```

### 2) RAP render-only run
```bash
python minimal_rap_bridge/render_pufferdrive_to_rap.py \
  --out-dir /tmp/pd_rap_render_native_aligned \
  --frames 80 \
  --episode-length 120 \
  --seed 1 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 32 \
  --cameras CAM_F0,CAM_L0,CAM_R0 \
  --ego-select-mode native_visualize \
  --include-ego-box \
  --replay-captured \
  --replay-source captured \
  --camera-yaw-mode heading
```

### 3) Change ego to inspect traffic context
```bash
--ego-select-mode index --ego-agent-index 10
```
Try `8,10,12,16,20` if ego ends up too far front/back.

## Files/Functions Most Relevant for Next Debug Step
- `minimal_rap_bridge/render_pufferdrive_to_rap.py`
- `choose_ego_index(...)`
- `resolve_ego_index_by_id(...)`
- `build_ground_truth_heading_lookup(...)`
- `apply_ground_truth_headings_to_state(...)`
- `extract_lane_polylines_abs_from_map_bin(...)`
- `build_boundary_map_features_static(...)`
- `run_bridge(...)` main integration loop

- `minimal_rap_bridge/compare_state_vs_rap.py`
- panel/metrics generation path
- map feature generation call
- replay state selection and heading correction path

- Native reference:
- `pufferlib/ocean/drive/visualize.c`
- `pufferlib/ocean/drive/drive.h`

## What To Verify Next (priority order)
- Verify if rotation is fully gone in long run with the latest heading correction.
- Verify road edges/lanes remain visible across frames after ego-relative static conversion.
- Compare same frame indices against native overview to check gross geometry alignment.
- If mismatch persists, align control/init settings exactly with native visualizer invocation and match selected human agent deterministically.

## Quick Triage Checklist
- Confirm script prints:
- `Ego select mode: ...`
- `Lane source polylines: ...`
- `Static boundary features: ...`
- Confirm map scenario:
- `Scenario id: ...` should stay consistent in single-map runs
- Confirm replay mode:
- `Replay mode: captured` or `ground_truth` exactly as intended

## Notes
- This handoff is a context snapshot for continuing in a new chat window.
- If you continue debugging, append date-stamped findings to this file to keep a running task log.

---

## Update 2026-02-14 (later session)

### New Root-Cause Findings
- `step_env_with_zeros(...)` was not a neutral/no-op rollout for discrete classic dynamics:
  - action `0` maps to max left steering + strong braking (`accel_idx=0`, `steer_idx=0`), which causes rapid heading drift/spin.
- `agent_id` is not unique in PufferDrive exports (both `get_global_agent_state()` and `get_ground_truth_trajectories()`):
  - using `{agent_id -> heading_series}` can assign wrong headings when duplicate IDs exist.
  - this can look like self-rotation artifacts after early frames.

### Code Fixes Applied
- Changed rollout stepping helper to use neutral controls:
  - classic discrete neutral action = `45` (`accel=0`, `steer=0`)
  - jerk discrete neutral action = `7` (`a_long=0`, `a_lat=0`)
  - retained function name `step_env_with_zeros(...)` for compatibility, but behavior is now neutral stepping.
- Reworked GT heading correction to be slot-stable:
  - `build_ground_truth_heading_lookup(...)` now maps by state slot index (not raw id), matching each slot to the best GT row once.
  - `apply_ground_truth_headings_to_state(...)` now applies by slot index.
- Improved ego tracking when duplicate IDs exist:
  - `resolve_ego_index_by_id(...)` now supports `last_ego_xy` and picks nearest candidate among same-id rows.
- Added duplicate box suppression in `build_anns_from_state(...)`:
  - suppresses exact overlapping duplicates from repeated active rows.

### Files Changed
- `minimal_rap_bridge/render_pufferdrive_to_rap.py`
- `minimal_rap_bridge/compare_state_vs_rap.py`

## Update 2026-02-14 (lateral orientation fix)

### New Finding
- Native-vs-RAP comparisons showed a consistent left/right inversion (ego lane side mirrored).
- Quantitative sanity check across 27 frames showed native alignment improved when RAP output was horizontally mirrored.

### Fix Applied
- Added `--flip-lateral-axis` flag to:
  - `minimal_rap_bridge/render_pufferdrive_to_rap.py`
  - `minimal_rap_bridge/compare_state_vs_rap.py`
- When enabled, bridge now flips lateral geometry consistently before RAP rendering:
  - map features: `y -> -y`
  - agent boxes: `rel_y -> -rel_y`, `yaw -> -yaw`
  - camera yaw: `ego_heading -> -ego_heading`

### Repro (native alignment test)
- RAP render:
  - `... render_pufferdrive_to_rap.py ... --ego-agent-index 359 --flip-lateral-axis`
- Compare:
  - `... compare_native_video_to_rap.py --native-video /tmp/pd_native_vs_rap/native_agent.mp4 --rap-dir /tmp/pd_native_vs_rap/rap_frames_ego359_flip ...`

## Update 2026-02-15 (frame-6 ego jump root cause)

### New Finding
- The observed abrupt side switch around frame 6 is a replay-source artifact, not just lateral flip:
  - with `--replay-source ground_truth`, selected ego slot/index (`359`) becomes invalid from frame 6 onward for this scene.
  - previous behavior re-resolved ego by duplicate id and jumped to another agent index (`0`), causing camera/side discontinuity.

### Fix Applied
- For `ground_truth` replay, ego slot is now locked (`Ego slot lock: True`) and no per-frame ego re-resolution is performed.
- If locked ego becomes invalid, camera pose is anchored to last valid ego pose instead of switching agents.

### Practical Recommendation
- For direct native-video comparison on this map, use:
  - `--replay-source captured --flip-lateral-axis`
- Reason:
  - `captured` keeps ego slot `359` valid and continuous across all tested frames;
  - `ground_truth` for this ego track ends early, so long-horizon native alignment is inherently unstable.

## Update 2026-02-15 (successful native-vs-RAP compare: noflip + captured)

### What Worked
This run produced a clean, stable side-by-side comparison between:
- Native PufferDrive `./visualize` agent video (already exported): `/tmp/pd_native_vs_rap/native_agent.mp4`
- RAP bridge frames rendered from PufferDrive with `captured` replay (no lateral flip): `/tmp/pd_native_vs_rap/rap_frames_ego359_noflip_captured`

Command:
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
- pairs: `/tmp/pd_native_vs_rap/compare_ego359_noflip_captured/pairs`
- overview: `/tmp/pd_native_vs_rap/compare_ego359_noflip_captured/overview.jpg`
- metrics: `/tmp/pd_native_vs_rap/compare_ego359_noflip_captured/metrics.csv`

### Interpretation Notes
- Use this primarily to validate *ego identity* + *left/right* + *gross neighborhood geometry*.
- Do not expect mid-horizon actor interactions to match perfectly unless the motion source is matched (see MVP limitations below).

### MVP Limitations (current bridge)
- Motion source mismatch:
  - Native `./visualize` is policy closed-loop: per-step `forward(net, obs) -> actions`, then `c_step`.
  - Bridge is not running that policy path; it uses `env.step(actions)` with either neutral actions or replayed actions/states.
  - Result: even with frame-0 alignment, neighbor trajectories can diverge by frame ~10-30.
- Camera mismatch is expected:
  - Native `--view agent` is a chase camera.
  - RAP `CAM_F0` is a fixed sensor-style camera model.
  - Match on “what is left/right/front” and “which cars exist”, not pixel-parity.
- Identity caveats in PufferDrive exports:
  - `id` is not reliably unique; slot-index stability is required for per-agent lookups.
- Map fidelity differences:
  - RAP currently uses road edges + parsed lane/line polylines; native draws richer road surface geometry.
  - Expect differences in how lanes/shoulders/edges look, even if geometry is roughly aligned.
