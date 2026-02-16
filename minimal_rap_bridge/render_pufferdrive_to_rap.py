#!/usr/bin/env python3
"""Minimal PufferDrive -> RAP renderer bridge (no C changes).

This script validates an end-to-end path using currently exposed Python APIs:
1) Step PufferDrive
2) Read global agent state + road edges
3) Convert to RAP ScenarioRenderer schema
4) Save rendered camera frames
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RAP_ROOT = REPO_ROOT / "third_party" / "RAP"

# Ensure local imports resolve without editable installs.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RAP_ROOT) not in sys.path:
    sys.path.insert(0, str(RAP_ROOT))

from pufferlib.ocean.drive.drive import Drive  # noqa: E402
from process_data.helpers.renderer import ScenarioRenderer  # noqa: E402
import cv2  # noqa: E402


@dataclass
class BridgeConfig:
    out_dir: Path
    frames: int
    seed: int
    ego_agent_index: int
    include_ego_box: bool
    assumed_height: float
    cameras: List[str]
    render_width: int
    render_height: int
    map_dir: str
    num_agents: int
    num_maps: int
    control_mode: str
    init_mode: str
    episode_length: int
    map_radius: float
    agent_radius: float | None
    rescale_intrinsics: bool
    replay_mode: bool
    replay_source: str
    ego_select_mode: str
    camera_yaw_mode: str
    camera_yaw_fixed_rad: float
    flip_lateral_axis: bool
    control_source: str
    policy_path: str
    actions_log_path: Path | None


@dataclass
class QaAccumulator:
    nonzero_pixels: Dict[str, List[int]]
    map_feature_counts: List[int]
    box_counts: List[int]


def build_ground_truth_heading_lookup(
    env: Drive,
    frames: int,
    state_ref: Dict[str, np.ndarray] | None = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return {state_slot: (heading[T], valid[T])} for stable per-frame heading correction.

    PufferDrive ids are not guaranteed unique, so we align each state slot to the
    best-matching GT row once, then apply headings by slot index.
    """
    gt = env.get_ground_truth_trajectories()
    traj_len = int(gt["heading"].shape[-1])
    if frames > traj_len:
        frames = traj_len
    if state_ref is None:
        state_ref = env.get_global_agent_state()

    gt_ids = np.array(gt["id"][:, 0], dtype=np.int32)
    gt_x = np.array(gt["x"][:, 0, :frames], dtype=np.float32)
    gt_y = np.array(gt["y"][:, 0, :frames], dtype=np.float32)
    gt_heading = np.array(gt["heading"][:, 0, :frames], dtype=np.float32)
    gt_valid = np.array(gt["valid"][:, 0, :frames], dtype=np.int32) > 0

    rows_by_id: Dict[int, List[int]] = {}
    for row, agent_id in enumerate(gt_ids.tolist()):
        rows_by_id.setdefault(int(agent_id), []).append(row)

    lookup: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    slot_count = int(state_ref["id"].shape[0])
    for slot in range(slot_count):
        agent_id = int(state_ref["id"][slot])
        candidates = rows_by_id.get(agent_id, [])
        if not candidates:
            continue

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            sx = float(state_ref["x"][slot])
            sy = float(state_ref["y"][slot])
            sh = float(state_ref["heading"][slot])
            best_cost = float("inf")
            chosen = candidates[0]
            for row in candidates:
                gx = float(gt_x[row, 0])
                gy = float(gt_y[row, 0])
                gh = float(gt_heading[row, 0])
                # Prefer a valid t=0 row and nearest pose match.
                valid_penalty = 0.0 if bool(gt_valid[row, 0]) else 1e6
                pos_cost = (sx - gx) * (sx - gx) + (sy - gy) * (sy - gy)
                hdiff = wrap_angle_rad(sh - gh)
                heading_cost = hdiff * hdiff
                cost = valid_penalty + pos_cost + 4.0 * heading_cost
                if cost < best_cost:
                    best_cost = cost
                    chosen = row

        lookup[slot] = (gt_heading[chosen], gt_valid[chosen])
    return lookup


def apply_ground_truth_headings_to_state(
    state: Dict[str, np.ndarray],
    t: int,
    heading_lookup: Dict[int, Tuple[np.ndarray, np.ndarray]] | None,
) -> None:
    """Overwrite state headings with GT headings (if available) to avoid spin drift."""
    if heading_lookup is None:
        return
    slot_count = int(state["heading"].shape[0])
    for slot in range(slot_count):
        rec = heading_lookup.get(slot)
        if rec is None:
            continue
        h_series, v_series = rec
        if t >= h_series.shape[0]:
            continue
        if not bool(v_series[t]):
            continue
        state["heading"][slot] = float(h_series[t])


def parse_args() -> BridgeConfig:
    parser = argparse.ArgumentParser(description="Render RAP-style frames from live PufferDrive state")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/pufferdrive_rap_minimal"))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ego-agent-index", type=int, default=0)
    parser.add_argument("--include-ego-box", action="store_true")
    parser.add_argument("--assumed-height", type=float, default=1.6, help="Fallback H for gt_boxes_world")
    parser.add_argument(
        "--cameras",
        type=str,
        default="CAM_F0,CAM_L0,CAM_R0",
        help="Comma-separated camera ids from RAP renderer camera_params",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=1920,
        help="Render width; RAP camera intrinsics are native at 1920x1120",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=1120,
        help="Render height; RAP camera intrinsics are native at 1920x1120",
    )
    parser.add_argument("--map-dir", type=str, default="resources/drive/binaries")
    parser.add_argument("--num-agents", type=int, default=32)
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument(
        "--control-mode",
        type=str,
        default="control_agents",
        choices=["control_vehicles", "control_agents", "control_wosac", "control_sdc_only"],
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="create_all_valid",
        choices=["create_all_valid", "create_only_controlled"],
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=200,
        help="Must be >= frames to avoid frequent episode boundaries",
    )
    parser.add_argument(
        "--map-radius",
        type=float,
        default=120.0,
        help="Meters around ego to keep for map/agent rendering stability",
    )
    parser.add_argument(
        "--agent-radius",
        type=float,
        default=None,
        help="Meters around ego to keep agent boxes (default: no radius filtering)",
    )
    parser.add_argument(
        "--no-rescale-intrinsics",
        action="store_true",
        help="Do not rescale camera intrinsics when using non-native render size",
    )
    parser.add_argument(
        "--replay-captured",
        action="store_true",
        help="Capture a rollout first, then replay/render it deterministically",
    )
    parser.add_argument(
        "--replay-source",
        type=str,
        default="captured",
        choices=["captured", "ground_truth"],
        help="Replay source when replay mode is enabled",
    )
    parser.add_argument(
        "--ego-select-mode",
        type=str,
        default="index",
        choices=["index", "native_visualize"],
        help="Ego selection policy: explicit index or native visualize-style first rand() pick",
    )
    parser.add_argument(
        "--replay-ground-truth",
        action="store_true",
        help="Replay logged ground-truth trajectories instead of open-loop neutral controls",
    )
    parser.add_argument(
        "--camera-yaw-mode",
        type=str,
        default="heading",
        choices=["heading", "fixed"],
        help="How to set camera yaw per frame",
    )
    parser.add_argument(
        "--camera-yaw-fixed-rad",
        type=float,
        default=0.0,
        help="Camera yaw (rad) when --camera-yaw-mode fixed",
    )
    parser.add_argument(
        "--flip-lateral-axis",
        action="store_true",
        help="Flip Y axis (and yaw sign) before feeding RAP renderer to match native left/right orientation",
    )
    parser.add_argument(
        "--control-source",
        type=str,
        default="neutral_actions",
        choices=["neutral_actions", "native_policy"],
        help="When not in replay mode, step environment with neutral actions or native policy forward+c_step",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="resources/drive/puffer_drive_weights.bin",
        help="Native policy weights path used when --control-source native_policy",
    )
    parser.add_argument(
        "--actions-log",
        type=Path,
        default=None,
        help="Optional CSV path to log native-policy actions per transition",
    )
    args = parser.parse_args()
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not cameras:
        raise ValueError("At least one camera must be provided via --cameras")
    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.episode_length < args.frames:
        raise ValueError("--episode-length must be >= --frames")

    return BridgeConfig(
        out_dir=args.out_dir,
        frames=args.frames,
        seed=args.seed,
        ego_agent_index=args.ego_agent_index,
        include_ego_box=args.include_ego_box,
        assumed_height=args.assumed_height,
        cameras=cameras,
        render_width=args.render_width,
        render_height=args.render_height,
        map_dir=args.map_dir,
        num_agents=args.num_agents,
        num_maps=args.num_maps,
        control_mode=args.control_mode,
        init_mode=args.init_mode,
        episode_length=args.episode_length,
        map_radius=args.map_radius,
        agent_radius=args.agent_radius,
        rescale_intrinsics=(not args.no_rescale_intrinsics),
        replay_mode=(args.replay_captured or args.replay_ground_truth),
        replay_source=("ground_truth" if args.replay_ground_truth else args.replay_source),
        ego_select_mode=args.ego_select_mode,
        camera_yaw_mode=args.camera_yaw_mode,
        camera_yaw_fixed_rad=args.camera_yaw_fixed_rad,
        flip_lateral_axis=args.flip_lateral_axis,
        control_source=args.control_source,
        policy_path=args.policy_path,
        actions_log_path=args.actions_log,
    )


def extract_boundary_polylines_abs(
    road_edges: Dict[str, np.ndarray],
    scenario_id_filter: str | None = None,
) -> List[np.ndarray]:
    """Extract absolute-world boundary polylines from flattened road-edge arrays."""
    polylines_abs: List[np.ndarray] = []
    lengths = road_edges["lengths"]
    xs = road_edges["x"]
    ys = road_edges["y"]
    sids = road_edges["scenario_id"]

    pt_idx = 0
    for i, seg_len in enumerate(lengths):
        seg_len = int(seg_len)
        seg_sid = str(sids[i])
        x_seg = xs[pt_idx : pt_idx + seg_len]
        y_seg = ys[pt_idx : pt_idx + seg_len]
        pt_idx += seg_len

        if scenario_id_filter is not None and seg_sid != scenario_id_filter:
            continue
        if seg_len < 2:
            continue

        polyline = np.stack([x_seg, y_seg], axis=1).astype(np.float32)
        if np.isfinite(polyline).all():
            polylines_abs.append(polyline)

    return polylines_abs


def split_contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return [start, end) runs of True values with length >= 2."""
    runs: List[Tuple[int, int]] = []
    start = None
    for i, keep in enumerate(mask):
        if keep and start is None:
            start = i
        elif not keep and start is not None:
            if i - start >= 2:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= 2:
        runs.append((start, len(mask)))
    return runs


def build_boundary_map_features_for_ego(
    polylines_abs: Sequence[np.ndarray],
    ego_x: float,
    ego_y: float,
    map_radius: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build RAP boundary map_features in ego-relative coordinates for current frame."""
    features: Dict[str, Dict[str, np.ndarray]] = {}
    feat_idx = 0
    ego_xy = np.array([ego_x, ego_y], dtype=np.float32)

    for poly_abs in polylines_abs:
        rel = poly_abs - ego_xy
        if rel.shape[0] < 2:
            continue
        finite_mask = np.isfinite(rel).all(axis=1)
        if not np.any(finite_mask):
            continue

        dists = np.linalg.norm(rel, axis=1)
        keep = finite_mask & (dists <= map_radius)
        if keep.sum() < 2:
            continue

        for st, ed in split_contiguous_runs(keep):
            seg = rel[st:ed].astype(np.float32)
            if seg.shape[0] < 2:
                continue
            # Hard bound to avoid pathological projections producing huge ints in cv2.clipLine.
            seg = np.clip(seg, -map_radius, map_radius)
            features[f"boundary_{feat_idx:05d}"] = {"type": "BOUNDARY", "polyline": seg}
            feat_idx += 1

    return features


def build_boundary_map_features_static(
    polylines_abs: Sequence[np.ndarray],
    lane_polylines_abs: Sequence[np.ndarray] | None = None,
    clip_bound: float = 600.0,
    ego_x: float = 0.0,
    ego_y: float = 0.0,
    flip_lateral_axis: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build dense static-style map_features in ego-relative coordinates."""
    features: Dict[str, Dict[str, np.ndarray]] = {}
    feat_idx = 0
    bound = max(float(clip_bound), 50.0)
    ego_xy = np.array([ego_x, ego_y], dtype=np.float32)
    for poly_abs in polylines_abs:
        seg = np.array(poly_abs, dtype=np.float32) - ego_xy
        if seg.shape[0] < 2 or not np.isfinite(seg).all():
            continue
        if flip_lateral_axis:
            seg[:, 1] = -seg[:, 1]
        # Keep "full static" behavior but cap coordinates to avoid pathological
        # projections that can overflow OpenCV clipLine integer parsing.
        seg = np.clip(seg, -bound, bound)
        features[f"boundary_{feat_idx:05d}"] = {"type": "BOUNDARY", "polyline": seg}
        feat_idx += 1

    lane_idx = 0
    if lane_polylines_abs is not None:
        for poly_abs in lane_polylines_abs:
            seg = np.array(poly_abs, dtype=np.float32) - ego_xy
            if seg.shape[0] < 2 or not np.isfinite(seg).all():
                continue
            if flip_lateral_axis:
                seg[:, 1] = -seg[:, 1]
            seg = np.clip(seg, -bound, bound)
            # RAP renderer checks `'LANE' in ftype` and consumes key `'polygon'`.
            features[f"lane_{lane_idx:05d}"] = {"type": "LANE_CENTER", "polygon": seg}
            lane_idx += 1
    return features


def pick_scenario_id(road_edges: Dict[str, np.ndarray]) -> str | None:
    if len(road_edges["scenario_id"]) == 0:
        return None
    unique = [str(v) for v in np.unique(road_edges["scenario_id"])]
    unique = [u for u in unique if u and u != ""]  # drop empty strings
    if not unique:
        return None
    return unique[0]


def choose_scenario_id_for_ego_from_road_edges(
    road_edges: Dict[str, np.ndarray],
    ego_x: float,
    ego_y: float,
    map_radius: float,
) -> Tuple[str | None, List[np.ndarray], int, int]:
    """Pick scenario_id whose boundaries best overlap ego neighborhood.

    In multi-map mode, road_edge exports can contain several scenario IDs.
    We choose the one that yields the most ego-local map features at the
    current ego pose, which avoids mismatching ego state and map geometry.
    """
    unique_sids = sorted({str(v) for v in road_edges["scenario_id"] if str(v)})
    best_sid: str | None = None
    best_polylines: List[np.ndarray] = []
    best_feature_count = -1

    for sid in unique_sids:
        polylines = extract_boundary_polylines_abs(road_edges, scenario_id_filter=sid)
        if not polylines:
            continue
        features = build_boundary_map_features_for_ego(
            polylines_abs=polylines,
            ego_x=ego_x,
            ego_y=ego_y,
            map_radius=map_radius,
        )
        count = len(features)
        if count > best_feature_count:
            best_sid = sid
            best_polylines = polylines
            best_feature_count = count

    if best_sid is None:
        fallback_sid = pick_scenario_id(road_edges)
        if fallback_sid is None:
            return None, [], 0, len(unique_sids)
        return fallback_sid, extract_boundary_polylines_abs(road_edges, scenario_id_filter=fallback_sid), 0, len(unique_sids)

    return best_sid, best_polylines, best_feature_count, len(unique_sids)


def valid_agent_indices(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Return indices with plausible populated state entries."""
    length = state["length"]
    width = state["width"]
    finite = np.isfinite(state["x"]) & np.isfinite(state["y"]) & np.isfinite(state["heading"])
    size_ok = (length > 0.0) & (width > 0.0)
    mask = finite & size_ok
    return np.flatnonzero(mask)


def choose_ego_index(valid_idx: np.ndarray, preferred: int, mode: str) -> int:
    if len(valid_idx) == 0:
        raise RuntimeError("No valid agents found in current state.")
    if mode == "native_visualize":
        # visualize.c uses: random_agent_idx = rand() % active_agent_count (no srand call).
        # On glibc default seed path, the first rand() is 1804289383.
        k = 1804289383 % len(valid_idx)
        return int(valid_idx[k])
    if preferred in set(valid_idx.tolist()):
        return preferred
    return int(valid_idx[0])


def infer_map_bin_path(env: Drive, map_dir: str) -> Path | None:
    map_ids = np.array(getattr(env, "map_ids", []), dtype=np.int32).reshape(-1)
    if map_ids.size > 0:
        candidate = Path(map_dir) / f"map_{int(map_ids[0]):03d}.bin"
        if candidate.exists():
            return candidate
    files = sorted(Path(map_dir).glob("map_*.bin"))
    if files:
        return files[0]
    return None


def _read_i32(f) -> int:
    data = f.read(4)
    if len(data) != 4:
        raise EOFError("unexpected EOF while reading int32")
    return struct.unpack("i", data)[0]


def extract_lane_polylines_abs_from_map_bin(map_bin_path: Path) -> List[np.ndarray]:
    """Extract ROAD_LANE / ROAD_LINE polylines from PufferDrive map binary."""
    lane_polylines: List[np.ndarray] = []
    with open(map_bin_path, "rb") as f:
        _ = f.read(16)  # scenario_id
        _ = _read_i32(f)  # sdc_track_index
        num_tracks_to_predict = _read_i32(f)
        if num_tracks_to_predict > 0:
            f.seek(4 * num_tracks_to_predict, os.SEEK_CUR)

        num_objects = _read_i32(f)
        num_roads = _read_i32(f)

        for _ in range(num_objects):
            _ = _read_i32(f)  # map_id
            _ = _read_i32(f)  # type
            _ = _read_i32(f)  # id
            arr = _read_i32(f)
            if arr < 0:
                raise ValueError(f"Invalid object array size in map binary: {arr}")
            # object payload: (x,y,z,vx,vy,vz,heading,valid) arrays + 7 scalar fields
            f.seek(8 * arr * 4 + 7 * 4, os.SEEK_CUR)

        ROAD_LANE = 4
        ROAD_LINE = 5
        for _ in range(num_roads):
            _ = _read_i32(f)  # map_id
            rtype = _read_i32(f)
            _ = _read_i32(f)  # id
            size = _read_i32(f)
            if size < 2:
                f.seek((3 * max(size, 0)) * 4 + 7 * 4, os.SEEK_CUR)
                continue
            x = np.fromfile(f, dtype=np.float32, count=size)
            y = np.fromfile(f, dtype=np.float32, count=size)
            _ = np.fromfile(f, dtype=np.float32, count=size)  # z
            f.seek(7 * 4, os.SEEK_CUR)
            if x.size != size or y.size != size:
                break
            if rtype in (ROAD_LANE, ROAD_LINE):
                poly = np.stack([x, y], axis=1).astype(np.float32)
                if np.isfinite(poly).all():
                    lane_polylines.append(poly)

    return lane_polylines


def resolve_ego_index_by_id(
    state: Dict[str, np.ndarray],
    ego_id: int,
    preferred_fallback: int,
    last_ego_xy: Tuple[float, float] | None = None,
) -> int:
    """Resolve current ego array index by stable agent id, with safe fallback."""
    valid_idx = valid_agent_indices(state)
    if len(valid_idx) == 0:
        raise RuntimeError("No valid agents available while resolving ego index")
    valid_set = set(valid_idx.tolist())

    idx_by_id = [int(i) for i in np.flatnonzero(state["id"] == ego_id).tolist() if int(i) in valid_set]
    if idx_by_id:
        if preferred_fallback in idx_by_id:
            return int(preferred_fallback)
        if last_ego_xy is not None and len(idx_by_id) > 1:
            ex, ey = last_ego_xy
            d2 = [
                (float(state["x"][i]) - ex) * (float(state["x"][i]) - ex)
                + (float(state["y"][i]) - ey) * (float(state["y"][i]) - ey)
                for i in idx_by_id
            ]
            return int(idx_by_id[int(np.argmin(np.array(d2, dtype=np.float32)))])
        return int(idx_by_id[0])

    if preferred_fallback in valid_set:
        return int(preferred_fallback)
    return int(valid_idx[0])


def wrap_angle_rad(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def compute_camera_yaw(raw_heading: float, cfg: BridgeConfig) -> float:
    if cfg.camera_yaw_mode == "fixed":
        yaw = wrap_angle_rad(cfg.camera_yaw_fixed_rad)
    else:
        yaw = wrap_angle_rad(raw_heading)
    if getattr(cfg, "flip_lateral_axis", False):
        yaw = wrap_angle_rad(-yaw)
    return yaw


def build_anns_from_state(
    state: Dict[str, np.ndarray],
    ego_idx: int,
    include_ego_box: bool,
    assumed_height: float,
    agent_radius: float | None,
    flip_lateral_axis: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create RAP anns fields from PufferDrive global state."""
    idxs = valid_agent_indices(state)
    if len(idxs) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=object)

    ego_x, ego_y, ego_z = state["x"][ego_idx], state["y"][ego_idx], state["z"][ego_idx]
    boxes: List[List[float]] = []
    names: List[str] = []
    seen_keys: set[Tuple[int, int, int, int, int]] = set()
    for i in idxs:
        if not include_ego_box and i == ego_idx:
            continue
        rel_x = float(state["x"][i] - ego_x)
        rel_y = float(state["y"][i] - ego_y)
        rel_z = float(state["z"][i] - ego_z)
        if flip_lateral_axis:
            rel_y = -rel_y
        if agent_radius is not None and agent_radius > 0.0:
            if (rel_x * rel_x + rel_y * rel_y) > agent_radius * agent_radius:
                continue
        agent_id = int(state["id"][i])
        length = float(state["length"][i])
        width = float(state["width"][i])
        # PufferDrive can emit duplicate active rows for some ids; suppress exact overlaps.
        key = (
            agent_id,
            int(round(rel_x * 100.0)),
            int(round(rel_y * 100.0)),
            int(round(length * 100.0)),
            int(round(width * 100.0)),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        yaw = float(state["heading"][i])  # keep world yaw, matching RAP metadata convention
        if flip_lateral_axis:
            yaw = -yaw
        boxes.append([rel_x, rel_y, rel_z, length, width, float(assumed_height), yaw])
        names.append("vehicle")

    if not boxes:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=object)
    return np.array(boxes, dtype=np.float32), np.array(names, dtype=object)


def make_scenario(
    state: Dict[str, np.ndarray],
    map_features: Dict[str, Dict[str, np.ndarray]],
    ego_idx: int,
    include_ego_box: bool,
    assumed_height: float,
    agent_radius: float | None,
    ego_heading: float,
    flip_lateral_axis: bool = False,
) -> Dict:
    gt_boxes_world, gt_names = build_anns_from_state(
        state=state,
        ego_idx=ego_idx,
        include_ego_box=include_ego_box,
        assumed_height=assumed_height,
        agent_radius=agent_radius,
        flip_lateral_axis=flip_lateral_axis,
    )
    return {
        "ego_heading": float(ego_heading),
        "traffic_lights": [],  # minimal path: no traffic light API yet in PufferDrive Python
        "map_features": map_features,
        "anns": {
            "gt_boxes_world": gt_boxes_world,
            "gt_names": gt_names,
        },
    }


def save_frame_images(out_dir: Path, frame_idx: int, rendered: Dict[str, np.ndarray]) -> None:
    for cam_id, image_rgb in rendered.items():
        out_path = out_dir / f"frame_{frame_idx:04d}_{cam_id}.jpg"
        # OpenCV expects BGR for file write
        cv2.imwrite(str(out_path), image_rgb[:, :, ::-1])


def get_nonzero_counts(rendered: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {cam: int((img > 0).sum()) for cam, img in rendered.items()}


def log_nonzero_counts(frame_idx: int, counts: Dict[str, int]) -> None:
    summary = ", ".join(f"{cam}={value}" for cam, value in counts.items())
    print(f"[frame {frame_idx:04d}] nonzero_pixels: {summary}")


def init_qa_accumulator(cameras: Sequence[str]) -> QaAccumulator:
    return QaAccumulator(
        nonzero_pixels={cam: [] for cam in cameras},
        map_feature_counts=[],
        box_counts=[],
    )


def maybe_rescale_renderer_intrinsics(
    renderer: ScenarioRenderer,
    width: int,
    height: int,
    enabled: bool,
    ref_width: int = 1920,
    ref_height: int = 1120,
) -> None:
    if not enabled:
        print("[info] intrinsics rescale: disabled")
        return
    sx = float(width) / float(ref_width)
    sy = float(height) / float(ref_height)
    for cam_id, cam_model in renderer.camera_models.items():
        K = np.array(cam_model["intrinsics"], dtype=np.float32, copy=True)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        cam_model["intrinsics"] = K
    print(
        f"[info] intrinsics rescale: enabled (sx={sx:.3f}, sy={sy:.3f}, "
        f"ref={ref_width}x{ref_height} -> render={width}x{height})"
    )


def update_qa_accumulator(
    qa: QaAccumulator,
    counts: Dict[str, int],
    map_feature_count: int,
    box_count: int,
) -> None:
    for cam, value in counts.items():
        qa.nonzero_pixels.setdefault(cam, []).append(int(value))
    qa.map_feature_counts.append(int(map_feature_count))
    qa.box_counts.append(int(box_count))


def summarize_series(series: Sequence[int]) -> Dict[str, float]:
    arr = np.array(series, dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def print_qa_summary(qa: QaAccumulator, frames: int) -> None:
    print("\n=== QA Summary ===")
    print(f"Frames rendered: {frames}")
    for cam in sorted(qa.nonzero_pixels.keys()):
        counts = qa.nonzero_pixels[cam]
        stats = summarize_series(counts)
        nonzero_frames = int(np.sum(np.array(counts) > 0))
        ratio = nonzero_frames / max(frames, 1)
        print(
            f"{cam}: nonzero_frames={nonzero_frames}/{frames} ({ratio:.1%}), "
            f"pixels[min/median/mean/max]="
            f"{int(stats['min'])}/{int(stats['median'])}/{int(stats['mean'])}/{int(stats['max'])}"
        )

    map_stats = summarize_series(qa.map_feature_counts)
    box_stats = summarize_series(qa.box_counts)
    print(
        "map_features_per_frame[min/median/mean/max]="
        f"{int(map_stats['min'])}/{int(map_stats['median'])}/{map_stats['mean']:.1f}/{int(map_stats['max'])}"
    )
    print(
        "agent_boxes_per_frame[min/median/mean/max]="
        f"{int(box_stats['min'])}/{int(box_stats['median'])}/{box_stats['mean']:.1f}/{int(box_stats['max'])}"
    )


def infer_neutral_discrete_action(env: Drive) -> int:
    """Return a neutral discrete action index for current dynamics model."""
    if getattr(env, "dynamics_model", "classic") == "jerk":
        # JERK_LONG[2] == 0, JERK_LAT[1] == 0 -> action = 2 * 3 + 1
        return 7
    # ACCEL[3] == 0, STEER[6] == 0 -> action = 3 * 13 + 6
    return 45


def build_neutral_actions(env: Drive) -> np.ndarray:
    actions = np.zeros_like(env.actions)
    if int(getattr(env, "_action_type_flag", 0)) == 0:
        actions.fill(infer_neutral_discrete_action(env))
    return actions


def step_env_with_zeros(env: Drive) -> None:
    # Backward-compatible name: this now uses neutral controls, not discrete action index 0.
    actions = build_neutral_actions(env)
    env.step(actions)


def copy_agent_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in state.items()}


def capture_rollout_states(env: Drive, frames: int) -> List[Dict[str, np.ndarray]]:
    states: List[Dict[str, np.ndarray]] = []
    for t in range(frames):
        states.append(copy_agent_state(env.get_global_agent_state()))
        if t < frames - 1:
            step_env_with_zeros(env)
    return states


def capture_ground_truth_rollout_states(env: Drive, frames: int) -> List[Dict[str, np.ndarray]]:
    """Build per-frame states from logged ground-truth trajectories.

    This avoids open-loop rollout drift/collisions when validating renderer conventions.
    """
    gt = env.get_ground_truth_trajectories()
    traj_len = int(gt["x"].shape[-1])
    if frames > traj_len:
        raise ValueError(f"Requested frames={frames}, but ground-truth trajectory length is {traj_len}")

    base = env.get_global_agent_state()
    base_id = np.array(base["id"], dtype=np.int32)
    base_len = np.array(base["length"], dtype=np.float32)
    base_wid = np.array(base["width"], dtype=np.float32)
    id_to_shape = {
        int(agent_id): (float(base_len[i]), float(base_wid[i]))
        for i, agent_id in enumerate(base_id.tolist())
    }

    gt_ids = np.array(gt["id"][:, 0], dtype=np.int32)
    n = gt_ids.shape[0]
    lengths = np.array([id_to_shape.get(int(agent_id), (4.5, 1.8))[0] for agent_id in gt_ids], dtype=np.float32)
    widths = np.array([id_to_shape.get(int(agent_id), (4.5, 1.8))[1] for agent_id in gt_ids], dtype=np.float32)

    states: List[Dict[str, np.ndarray]] = []
    for t in range(frames):
        valid = np.array(gt["valid"][:, 0, t], dtype=np.int32) > 0
        x = np.array(gt["x"][:, 0, t], dtype=np.float32)
        y = np.array(gt["y"][:, 0, t], dtype=np.float32)
        z = np.array(gt["z"][:, 0, t], dtype=np.float32)
        h = np.array(gt["heading"][:, 0, t], dtype=np.float32)

        # Drop invalid tracked points by NaN-ing pose; valid_agent_indices handles this.
        x[~valid] = np.nan
        y[~valid] = np.nan
        z[~valid] = np.nan
        h[~valid] = np.nan

        states.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "heading": h,
                "id": gt_ids.copy(),
                "length": lengths.copy(),
                "width": widths.copy(),
            }
        )

    return states


def run_bridge(cfg: BridgeConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    env = Drive(
        num_agents=cfg.num_agents,
        num_maps=cfg.num_maps,
        map_dir=cfg.map_dir,
        control_mode=cfg.control_mode,
        init_mode=cfg.init_mode,
        episode_length=cfg.episode_length,
        resample_frequency=0,  # keep scenario stable during minimal validation
        report_interval=max(cfg.frames, 1),
    )
    try:
        env.reset(seed=cfg.seed)
        if cfg.replay_mode and cfg.control_source == "native_policy":
            raise ValueError("native_policy control source cannot be combined with replay mode")
        if (not cfg.replay_mode) and cfg.control_source == "native_policy":
            env.init_native_policy(cfg.policy_path)
        state0 = env.get_global_agent_state()
        valid0 = valid_agent_indices(state0)
        ego_idx = choose_ego_index(valid0, cfg.ego_agent_index, cfg.ego_select_mode)
        ego_id = int(state0["id"][ego_idx])
        replay_states: List[Dict[str, np.ndarray]] | None = None
        gt_heading_lookup: Dict[int, Tuple[np.ndarray, np.ndarray]] | None = None
        if cfg.replay_mode:
            if cfg.replay_source == "ground_truth":
                replay_states = capture_ground_truth_rollout_states(env, cfg.frames)
            else:
                replay_states = capture_rollout_states(env, cfg.frames)
                gt_heading_lookup = build_ground_truth_heading_lookup(env, cfg.frames, state_ref=replay_states[0])
            state_ref = replay_states[0]
        else:
            state_ref = state0

        road_edges = env.get_road_edge_polylines()
        sid, boundary_polylines_abs, initial_feature_count, num_sid_candidates = choose_scenario_id_for_ego_from_road_edges(
            road_edges=road_edges,
            ego_x=float(state_ref["x"][ego_idx]),
            ego_y=float(state_ref["y"][ego_idx]),
            map_radius=cfg.map_radius,
        )
        if not boundary_polylines_abs:
            raise RuntimeError("No road-edge map features available; cannot build RAP map_features.")

        renderer = ScenarioRenderer(
            camera_channel_list=cfg.cameras,
            width=cfg.render_width,
            height=cfg.render_height,
        )
        maybe_rescale_renderer_intrinsics(
            renderer=renderer,
            width=cfg.render_width,
            height=cfg.render_height,
            enabled=cfg.rescale_intrinsics,
        )

        print(f"Output directory: {cfg.out_dir}")
        print(f"Selected ego index: {ego_idx}")
        print(f"Selected ego id: {ego_id}")
        print(f"Ego select mode: {cfg.ego_select_mode}")
        print(f"Scenario id: {sid}")
        print(f"Boundary source polylines: {len(boundary_polylines_abs)}")
        print(f"Scenario id candidates in road edges: {num_sid_candidates}")
        print(f"Initial map feature count near ego: {initial_feature_count}")
        if cfg.replay_mode:
            print(f"Replay mode: {cfg.replay_source}")
        else:
            print(f"Replay mode: off (control_source={cfg.control_source})")
            if cfg.control_source == "native_policy":
                print(f"Policy path: {cfg.policy_path}")
        print(
            f"Camera yaw mode: {cfg.camera_yaw_mode} (fixed_rad={cfg.camera_yaw_fixed_rad:.3f})"
        )
        print(f"Flip lateral axis: {cfg.flip_lateral_axis}")
        print(f"Radius settings: map_radius={cfg.map_radius:.1f}m, agent_radius={cfg.agent_radius}")
        print(f"Cameras: {cfg.cameras}")
        lock_ego_slot = bool(cfg.replay_mode and cfg.replay_source == "ground_truth")
        print(f"Ego slot lock: {lock_ego_slot}")
        qa = init_qa_accumulator(cfg.cameras)
        action_log_file = None
        action_log_writer = None
        if cfg.actions_log_path is not None:
            cfg.actions_log_path.parent.mkdir(parents=True, exist_ok=True)
            action_log_file = open(cfg.actions_log_path, "w", newline="", encoding="utf-8")
            action_log_writer = csv.DictWriter(
                action_log_file,
                fieldnames=[
                    "frame",
                    "ego_slot",
                    "ego_id",
                    "ego_action",
                    "action_count",
                    "action_min",
                    "action_max",
                    "action_sum",
                    "action_crc32",
                ],
            )
            action_log_writer.writeheader()
            print(f"Actions log: {cfg.actions_log_path}")
        map_bin_path = infer_map_bin_path(env, cfg.map_dir)
        lane_polylines_abs: List[np.ndarray] = []
        if map_bin_path is not None:
            try:
                lane_polylines_abs = extract_lane_polylines_abs_from_map_bin(map_bin_path)
            except Exception as e:
                print(f"[warn] failed to parse lane polylines from {map_bin_path}: {e}")
        print(f"Lane source polylines: {len(lane_polylines_abs)}")
        current_ego_idx = resolve_ego_index_by_id(state_ref, ego_id=ego_id, preferred_fallback=ego_idx)
        last_ego_x = float(state_ref["x"][current_ego_idx])
        last_ego_y = float(state_ref["y"][current_ego_idx])
        last_ego_heading = float(state_ref["heading"][current_ego_idx])
        static_clip_bound = max(cfg.map_radius * 5.0, 200.0)
        static_map_features_preview = build_boundary_map_features_static(
            boundary_polylines_abs,
            lane_polylines_abs=lane_polylines_abs,
            clip_bound=static_clip_bound,
            ego_x=last_ego_x,
            ego_y=last_ego_y,
            flip_lateral_axis=cfg.flip_lateral_axis,
        )
        print(f"Static boundary features: {len(static_map_features_preview)}")

        for t in range(cfg.frames):
            if cfg.replay_mode:
                assert replay_states is not None
                state = replay_states[t]
                apply_ground_truth_headings_to_state(state, t, gt_heading_lookup)
            else:
                state = env.get_global_agent_state()
            if not lock_ego_slot:
                current_ego_idx = resolve_ego_index_by_id(
                    state,
                    ego_id=ego_id,
                    preferred_fallback=current_ego_idx,
                    last_ego_xy=(last_ego_x, last_ego_y),
                )

            if np.isfinite(state["x"][current_ego_idx]) and np.isfinite(state["y"][current_ego_idx]) and np.isfinite(
                state["heading"][current_ego_idx]
            ):
                ego_x = float(state["x"][current_ego_idx])
                ego_y = float(state["y"][current_ego_idx])
                raw_heading = float(state["heading"][current_ego_idx])
                last_ego_x, last_ego_y, last_ego_heading = ego_x, ego_y, raw_heading
            else:
                # Keep camera anchored if the chosen ego is temporarily invalid in replay.
                ego_x, ego_y, raw_heading = last_ego_x, last_ego_y, last_ego_heading
                state["x"][current_ego_idx] = ego_x
                state["y"][current_ego_idx] = ego_y
                state["heading"][current_ego_idx] = raw_heading

            camera_yaw = compute_camera_yaw(raw_heading, cfg)

            map_features = build_boundary_map_features_static(
                boundary_polylines_abs,
                lane_polylines_abs=lane_polylines_abs,
                clip_bound=static_clip_bound,
                ego_x=ego_x,
                ego_y=ego_y,
                flip_lateral_axis=cfg.flip_lateral_axis,
            )
            scenario = make_scenario(
                state=state,
                map_features=map_features,
                ego_idx=current_ego_idx,
                include_ego_box=cfg.include_ego_box,
                assumed_height=cfg.assumed_height,
                agent_radius=cfg.agent_radius,
                ego_heading=camera_yaw,
                flip_lateral_axis=cfg.flip_lateral_axis,
            )
            rendered = renderer.observe(scenario)
            save_frame_images(cfg.out_dir, t, rendered)
            nonzero_counts = get_nonzero_counts(rendered)
            log_nonzero_counts(t, nonzero_counts)
            update_qa_accumulator(
                qa=qa,
                counts=nonzero_counts,
                map_feature_count=len(map_features),
                box_count=int(scenario["anns"]["gt_boxes_world"].shape[0]),
            )
            if (not cfg.replay_mode) and (t < cfg.frames - 1):
                step_ego_slot = int(current_ego_idx)
                step_ego_id = int(state["id"][step_ego_slot]) if step_ego_slot < state["id"].shape[0] else -1
                if cfg.control_source == "native_policy":
                    env.step_native_policy()
                    if action_log_writer is not None:
                        acts = np.array(env.actions, dtype=np.int32, copy=False).reshape(-1)
                        ego_action = int(acts[step_ego_slot]) if 0 <= step_ego_slot < acts.shape[0] else -1
                        action_log_writer.writerow(
                            {
                                "frame": t,
                                "ego_slot": step_ego_slot,
                                "ego_id": step_ego_id,
                                "ego_action": ego_action,
                                "action_count": int(acts.shape[0]),
                                "action_min": int(np.min(acts)) if acts.size else 0,
                                "action_max": int(np.max(acts)) if acts.size else 0,
                                "action_sum": int(np.sum(acts, dtype=np.int64)) if acts.size else 0,
                                "action_crc32": int(zlib.crc32(acts.tobytes()) & 0xFFFFFFFF),
                            }
                        )
                else:
                    step_env_with_zeros(env)
        print_qa_summary(qa, cfg.frames)

    finally:
        if "action_log_file" in locals() and action_log_file is not None:
            action_log_file.close()
        env.close()


def main() -> None:
    cfg = parse_args()
    run_bridge(cfg)


if __name__ == "__main__":
    main()
