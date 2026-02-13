# PufferDrive -> RAP Renderer Integration Spec

This document organizes:

1. What `ScenarioRenderer` needs (exact schema/assumptions)
2. What PufferDrive exposes today (Python API)
3. What data exists internally in PufferDrive C core (not yet exported)
4. Mapping/gap analysis for building a pipeline

---

## 1) RAP `ScenarioRenderer` Input Contract

Renderer entrypoint:

- `third_party/RAP/process_data/helpers/renderer.py:704`

Hard-required top-level keys in `scenario`:

- `scenario["ego_heading"]` used at `third_party/RAP/process_data/helpers/renderer.py:706`
- `scenario["traffic_lights"]` iterated at `third_party/RAP/process_data/helpers/renderer.py:717`
- `scenario["map_features"]` iterated at `third_party/RAP/process_data/helpers/renderer.py:734`
- `scenario["anns"]` read at `third_party/RAP/process_data/helpers/renderer.py:755`

Expected nested fields:

- `traffic_lights`: sequence of tuples where renderer reads:
  - `feat[1] -> is_red`
  - `feat[2] -> [x, y]`
  - See `third_party/RAP/process_data/helpers/renderer.py:718`
- `map_features`: dict-like; each feature has:
  - `feat["type"]`
  - if `"LANE"` in type: `feat["polygon"]`
  - if `"CROSSWALK"` or `"SPEED_BUMP"` in type: `feat["polygon"]`
  - if `"BOUNDARY"` or `"SOLID"` in type: `feat["polyline"]`
  - See `third_party/RAP/process_data/helpers/renderer.py:735`
- `anns`:
  - `anns["gt_boxes_world"]` shape `(N, >=7)` with `[x, y, z, L, W, H, yaw, ...]`
    - Explicit in `third_party/RAP/process_data/helpers/renderer.py:365`
  - `anns["gt_names"]` key exists (currently loaded but not used for color logic)
    - `third_party/RAP/process_data/helpers/renderer.py:757`

Coordinate convention in RAP metadata pipeline:

- `create_openscene_metadata.py` computes `gt_boxes_world` as object position minus ego position; yaw is world yaw:
  - `third_party/RAP/process_data/create_openscene_metadata.py:430`
  - `third_party/RAP/process_data/create_openscene_metadata.py:436`
- `scenario["ego_heading"]` is ego world yaw:
  - `third_party/RAP/process_data/create_openscene_metadata.py:301`
- `scenario["map_features"]` and `traffic_lights` are built around ego-centered map query and relative conversion:
  - `third_party/RAP/process_data/create_openscene_metadata.py:298`
  - `third_party/RAP/process_data/create_openscene_metadata.py:287`
- `scenario["ego_pos"]` is set in metadata scripts but not used inside renderer:
  - set at `third_party/RAP/process_data/create_openscene_metadata.py:300`
  - renderer uses only `ego_heading` + `lidar_pos = zeros(3)` at `third_party/RAP/process_data/helpers/renderer.py:705`

---

## 2) PufferDrive Data Exposed Today (Python Layer)

Environment wrapper:

- `pufferlib/ocean/drive/drive.py`

Main rollout/state APIs:

- `step(actions)` returns:
  - `observations, rewards, terminals, truncations, info`
  - `pufferlib/ocean/drive/drive.py:271`
- `get_global_agent_state()` returns arrays:
  - `x, y, z, heading, id, length, width`
  - `pufferlib/ocean/drive/drive.py:287`
- `get_ground_truth_trajectories()` returns arrays:
  - `x, y, z, heading, valid, id, is_vehicle, is_track_to_predict, scenario_id`
  - `pufferlib/ocean/drive/drive.py:319`
- `get_road_edge_polylines()` returns:
  - `x, y, lengths, scenario_id`
  - `pufferlib/ocean/drive/drive.py:359`

Observation tensor composition:

- Observation size/config in `Drive.__init__`:
  - `pufferlib/ocean/drive/drive.py:67`
- C-side computed features:
  - Ego features incl. goal rel pos, speed, size, collision flag; jerk extras if enabled
  - Partner features: rel x/y, size, rel heading, speed
  - Road features: segment rel x/y, length, width, direction, type
  - `pufferlib/ocean/drive/drive.h:1750`
  - `pufferlib/ocean/drive/drive.h:1901`

Bindings exported to Python module:

- method table includes:
  - global state, GT trajectories, road edges
  - `pufferlib/ocean/env_binding.h:977`

---

## 3) PufferDrive Full Internal Data (C Core)

Entity/road/object types available in sim:

- `VEHICLE`, `PEDESTRIAN`, `CYCLIST`, `ROAD_LANE`, `ROAD_LINE`, `ROAD_EDGE`, `STOP_SIGN`, `CROSSWALK`, `SPEED_BUMP`, `DRIVEWAY`
- `pufferlib/ocean/drive/drive.h:17`

`Entity` has rich per-agent/per-road fields:

- IDs/types, full trajectories (`traj_x/y/z`, `traj_vx/vy/vz`, `traj_heading`, `traj_valid`), dimensions (`width/length/height`), goal fields, dynamics state, collision/metrics flags, etc.
- `pufferlib/ocean/drive/drive.h:169`

`Drive` has scenario/global metadata:

- active/static agent indices, world offsets, scenario id, tracks_to_predict, control/init modes, etc.
- `pufferlib/ocean/drive/drive.h:280`

Map binary loads all objects + road entities with scalar + trajectory arrays:

- loader:
  - `pufferlib/ocean/drive/drive.h:380`
- includes:
  - `scenario_id`, `sdc_track_index`, `tracks_to_predict`, per-entity type/id/geometry/size/goals
  - `pufferlib/ocean/drive/drive.h:385`
  - `pufferlib/ocean/drive/drive.h:447`

Current C extractors implemented:

- `c_get_global_agent_state`
  - `pufferlib/ocean/drive/drive.h:1676`
- `c_get_global_ground_truth_trajectories`
  - `pufferlib/ocean/drive/drive.h:1693`
- `c_get_road_edge_polylines` (road-edge only)
  - `pufferlib/ocean/drive/drive.h:1730`

Important: more data exists internally than is currently exported via Python API.

---

## 4) Mapping to RAP Renderer: Ready vs Missing

### Ready now (no new bindings)

- `scenario["ego_heading"]`
  - from chosen ego agent `heading` in `get_global_agent_state()`
- `scenario["anns"]["gt_boxes_world"]` partially
  - can fill `[x, y, z, L, W, ?, yaw]` from `get_global_agent_state()`
  - missing height (`H`) unless you use a fixed constant
- `scenario["map_features"]` partial
  - can create `"BOUNDARY"` entries from `get_road_edge_polylines()`
- `scenario["traffic_lights"]`
  - can set empty list for now

### Missing for full parity with RAP data pipeline

- Per-agent `height` and `type/name` in global state API
  - exists in `Entity`, not returned by current Python API
  - `pufferlib/ocean/drive/drive.h:182`
- Full map layers for renderer (`LANE`, `CROSSWALK`, `SPEED_BUMP`, `SOLID`) via Python API
  - all road entity types exist internally
  - currently only road edges are exported
- Traffic light states/positions
  - RAP metadata builds them from nuPlan DB; PufferDrive map binary path does not currently expose an equivalent API

---

## 5) Recommended Integration Plan

### Phase A: Minimal working bridge (start here)

1. Build `scenario["anns"]["gt_boxes_world"]` from `get_global_agent_state()`, with fixed height (e.g., `1.6`).
2. Build `scenario["map_features"]` using road-edge polylines as `BOUNDARY`.
3. Set `scenario["traffic_lights"] = []`.
4. Render with `ScenarioRenderer.observe`.

This is enough to validate end-to-end PufferDrive -> RAP raster rendering.

### Phase B: Better fidelity bindings

Add new binding(s) to export:

1. Agent extras: `height`, `type`, optional `vx/vy`, collision state.
2. General road geometry extraction by type (`ROAD_LANE`, `ROAD_LINE`, `CROSSWALK`, `SPEED_BUMP`, `DRIVEWAY`, etc.), not only road edge.
3. Optional goal/state extras for analysis overlays.

---

## 6) Practical Frame Schema for Adapter

Suggested canonical per-frame structure before RAP conversion:

```python
{
  "ego": {"agent_id": int, "heading": float},
  "agents": {
    "id": np.ndarray[N],
    "x": np.ndarray[N], "y": np.ndarray[N], "z": np.ndarray[N],
    "heading": np.ndarray[N],
    "length": np.ndarray[N], "width": np.ndarray[N],
    # optional future: height, type, vx, vy, collision_state
  },
  "map": {
    "road_edges": {
      "x": np.ndarray[M], "y": np.ndarray[M], "lengths": np.ndarray[K], "scenario_id": np.ndarray[K]
    }
    # optional future: lanes/crosswalks/speedbumps/lines by type
  },
  "obs": np.ndarray[num_agents, num_obs],  # from step()
}
```

Then convert this canonical frame into:

```python
scenario = {
  "ego_heading": ...,
  "traffic_lights": ...,
  "map_features": ...,
  "anns": {"gt_boxes_world": ..., "gt_names": ...},
}
```

