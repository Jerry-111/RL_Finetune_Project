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
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
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
    )


def build_boundary_map_features(
    road_edges: Dict[str, np.ndarray],
    scenario_id_filter: str | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Convert flattened road-edge output into RAP map_features entries."""
    features: Dict[str, Dict[str, np.ndarray]] = {}
    lengths = road_edges["lengths"]
    xs = road_edges["x"]
    ys = road_edges["y"]
    sids = road_edges["scenario_id"]

    pt_idx = 0
    feat_idx = 0
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
        features[f"boundary_{feat_idx:05d}"] = {"type": "BOUNDARY", "polyline": polyline}
        feat_idx += 1

    return features


def pick_scenario_id(road_edges: Dict[str, np.ndarray]) -> str | None:
    if len(road_edges["scenario_id"]) == 0:
        return None
    unique = [str(v) for v in np.unique(road_edges["scenario_id"])]
    unique = [u for u in unique if u and u != ""]  # drop empty strings
    if not unique:
        return None
    return unique[0]


def valid_agent_indices(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Return indices with plausible populated state entries."""
    length = state["length"]
    width = state["width"]
    finite = np.isfinite(state["x"]) & np.isfinite(state["y"]) & np.isfinite(state["heading"])
    size_ok = (length > 0.0) & (width > 0.0)
    mask = finite & size_ok
    return np.flatnonzero(mask)


def choose_ego_index(valid_idx: np.ndarray, preferred: int) -> int:
    if len(valid_idx) == 0:
        raise RuntimeError("No valid agents found in current state.")
    if preferred in set(valid_idx.tolist()):
        return preferred
    return int(valid_idx[0])


def build_anns_from_state(
    state: Dict[str, np.ndarray],
    ego_idx: int,
    include_ego_box: bool,
    assumed_height: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create RAP anns fields from PufferDrive global state."""
    idxs = valid_agent_indices(state)
    if len(idxs) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=object)

    ego_x, ego_y, ego_z = state["x"][ego_idx], state["y"][ego_idx], state["z"][ego_idx]
    boxes: List[List[float]] = []
    names: List[str] = []
    for i in idxs:
        if not include_ego_box and i == ego_idx:
            continue
        rel_x = float(state["x"][i] - ego_x)
        rel_y = float(state["y"][i] - ego_y)
        rel_z = float(state["z"][i] - ego_z)
        length = float(state["length"][i])
        width = float(state["width"][i])
        yaw = float(state["heading"][i])  # keep world yaw, matching RAP metadata convention
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
) -> Dict:
    gt_boxes_world, gt_names = build_anns_from_state(
        state=state,
        ego_idx=ego_idx,
        include_ego_box=include_ego_box,
        assumed_height=assumed_height,
    )
    return {
        "ego_heading": float(state["heading"][ego_idx]),
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


def log_nonzero_counts(frame_idx: int, rendered: Dict[str, np.ndarray]) -> None:
    counts = ", ".join(f"{cam}={int((img > 0).sum())}" for cam, img in rendered.items())
    print(f"[frame {frame_idx:04d}] nonzero_pixels: {counts}")


def step_env_with_zeros(env: Drive) -> None:
    actions = np.zeros_like(env.actions)
    env.step(actions)


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
        state0 = env.get_global_agent_state()
        valid0 = valid_agent_indices(state0)
        ego_idx = choose_ego_index(valid0, cfg.ego_agent_index)

        road_edges = env.get_road_edge_polylines()
        sid = pick_scenario_id(road_edges)
        map_features = build_boundary_map_features(road_edges, scenario_id_filter=sid)
        if not map_features:
            raise RuntimeError("No road-edge map features available; cannot build RAP map_features.")

        renderer = ScenarioRenderer(
            camera_channel_list=cfg.cameras,
            width=cfg.render_width,
            height=cfg.render_height,
        )

        print(f"Output directory: {cfg.out_dir}")
        print(f"Selected ego index: {ego_idx}")
        print(f"Boundary features: {len(map_features)}")
        print(f"Cameras: {cfg.cameras}")

        for t in range(cfg.frames):
            state = env.get_global_agent_state()
            scenario = make_scenario(
                state=state,
                map_features=map_features,
                ego_idx=ego_idx,
                include_ego_box=cfg.include_ego_box,
                assumed_height=cfg.assumed_height,
            )
            rendered = renderer.observe(scenario)
            save_frame_images(cfg.out_dir, t, rendered)
            log_nonzero_counts(t, rendered)
            step_env_with_zeros(env)

    finally:
        env.close()


def main() -> None:
    cfg = parse_args()
    run_bridge(cfg)


if __name__ == "__main__":
    main()
