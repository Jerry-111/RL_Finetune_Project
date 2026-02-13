# Minimal PufferDrive -> RAP Bridge

This folder provides a minimal viable validation path (Option 1) without changing PufferDrive C internals.

Script:

- `minimal_rap_bridge/render_pufferdrive_to_rap.py`

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
- `--include-ego-box`
- `--assumed-height 1.6`
- `--render-width 1280 --render-height 720`
- `--control-mode control_agents`
- `--init-mode create_all_valid`

---

## Output

Images are saved as:

- `/tmp/pufferdrive_rap_minimal/frame_0000_CAM_F0.jpg`
- `/tmp/pufferdrive_rap_minimal/frame_0000_CAM_L0.jpg`
- `/tmp/pufferdrive_rap_minimal/frame_0000_CAM_R0.jpg`
- ...

The script also prints per-frame nonzero pixel counts per camera as a quick sanity signal.
