#!/usr/bin/env python3
"""State-level bridge validation: PufferDrive state BEV vs RAP camera render.

This does not depend on native Drive pixel output APIs (not currently exposed).
It validates that extracted map/agent state produces plausible RAP inputs by
rendering:
1) a simple BEV panel directly from extracted state-derived scenario fields
2) RAP camera output for the same frame
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from minimal_rap_bridge.render_pufferdrive_to_rap import (
    Drive,
    ScenarioRenderer,
    apply_ground_truth_headings_to_state,
    build_ground_truth_heading_lookup,
    capture_ground_truth_rollout_states,
    build_boundary_map_features_static,
    capture_rollout_states,
    choose_ego_index,
    choose_scenario_id_for_ego_from_road_edges,
    compute_camera_yaw,
    extract_lane_polylines_abs_from_map_bin,
    get_nonzero_counts,
    infer_map_bin_path,
    make_scenario,
    maybe_rescale_renderer_intrinsics,
    resolve_ego_index_by_id,
    step_env_with_zeros,
    valid_agent_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PufferDrive state BEV panel vs RAP render")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/pd_rap_state_compare"))
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--map-dir", type=str, default="resources/drive/binaries")
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=32)
    parser.add_argument(
        "--cameras",
        type=str,
        default="CAM_F0,CAM_L0,CAM_R0",
        help="Comma-separated RAP camera ids",
    )
    parser.add_argument("--panel-camera", type=str, default="CAM_F0")
    parser.add_argument("--render-width", type=int, default=1920)
    parser.add_argument("--render-height", type=int, default=1120)
    parser.add_argument("--ego-agent-index", type=int, default=6)
    parser.add_argument(
        "--ego-select-mode",
        type=str,
        default="index",
        choices=["index", "native_visualize"],
        help="Ego selection policy: explicit index or native visualize-style first rand() pick",
    )
    parser.add_argument("--include-ego-box", action="store_true")
    parser.add_argument("--assumed-height", type=float, default=1.6)
    parser.add_argument("--map-radius", type=float, default=100.0)
    parser.add_argument(
        "--agent-radius",
        type=float,
        default=None,
        help="Meters around ego to keep agent boxes (default: no radius filtering)",
    )
    parser.add_argument("--bev-size", type=int, default=720, help="Square BEV panel size in pixels")
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
    parser.add_argument("--no-rescale-intrinsics", action="store_true")
    parser.add_argument("--replay-captured", action="store_true")
    parser.add_argument(
        "--replay-source",
        type=str,
        default="captured",
        choices=["captured", "ground_truth"],
        help="Replay source when --replay-captured is enabled",
    )
    parser.add_argument(
        "--camera-yaw-mode",
        type=str,
        default="heading",
        choices=["heading", "fixed"],
    )
    parser.add_argument("--camera-yaw-fixed-rad", type=float, default=0.0)
    parser.add_argument(
        "--show-native-window",
        action="store_true",
        help="Call Drive.render() each step for manual native inspection (requires display)",
    )
    parser.add_argument("--native-render-every", type=int, default=1)

    args = parser.parse_args()
    if args.episode_length < args.frames:
        raise ValueError("--episode-length must be >= --frames")
    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.native_render_every <= 0:
        raise ValueError("--native-render-every must be > 0")
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not cameras:
        raise ValueError("No cameras provided")
    if args.panel_camera not in cameras:
        raise ValueError(f"--panel-camera {args.panel_camera} must be in --cameras {cameras}")
    args.cameras = cameras
    return args


def world_to_px(x: float, y: float, size: int, radius: float) -> tuple[int, int]:
    scale = (0.45 * float(size)) / max(radius, 1e-6)
    cx = size * 0.5
    cy = size * 0.5
    px = int(round(cx + x * scale))
    py = int(round(cy - y * scale))
    return px, py


def box_corners_xy(cx: float, cy: float, length: float, width: float, yaw: float) -> np.ndarray:
    half_l = 0.5 * float(length)
    half_w = 0.5 * float(width)
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([cx, cy], dtype=np.float32)


def draw_bev_panel(scenario: Dict, size: int, radius: float, camera_yaw: float) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for feat in scenario["map_features"].values():
        key = "polyline" if "polyline" in feat else "polygon"
        pts = np.array(feat[key], dtype=np.float32)
        if pts.shape[0] < 2:
            continue
        px = [world_to_px(float(p[0]), float(p[1]), size, radius) for p in pts]
        px_arr = np.array(px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [px_arr], isClosed=False, color=(30, 40, 210), thickness=2, lineType=cv2.LINE_AA)

    boxes = np.array(scenario["anns"]["gt_boxes_world"], dtype=np.float32)
    for i, box in enumerate(boxes):
        cx, cy, _, length, width, _, yaw = box.tolist()
        corners = box_corners_xy(cx, cy, length, width, yaw)
        px = np.array([world_to_px(float(p[0]), float(p[1]), size, radius) for p in corners], dtype=np.int32)
        poly = px.reshape(-1, 1, 2)
        face_color = (190, 70, 230) if (i % 2 == 0) else (210, 170, 50)
        cv2.fillPoly(img, [poly], color=face_color, lineType=cv2.LINE_AA)
        cv2.polylines(img, [poly], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        front = np.mean(px[:2], axis=0).astype(np.int32)
        center = np.mean(px, axis=0).astype(np.int32)
        cv2.arrowedLine(img, tuple(center), tuple(front), color=(255, 255, 255), thickness=1, tipLength=0.3)

    # Ego origin + camera heading arrow.
    ego_px = world_to_px(0.0, 0.0, size, radius)
    cv2.circle(img, ego_px, 4, (30, 230, 30), thickness=-1, lineType=cv2.LINE_AA)
    arrow_len = 0.15 * radius
    hx = arrow_len * float(np.cos(camera_yaw))
    hy = arrow_len * float(np.sin(camera_yaw))
    head_px = world_to_px(hx, hy, size, radius)
    cv2.arrowedLine(img, ego_px, head_px, color=(30, 230, 30), thickness=2, tipLength=0.25)
    cv2.putText(img, "PufferDrive state BEV", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    return img


def compose_panel(bev_bgr: np.ndarray, rap_rgb: np.ndarray, label: str) -> np.ndarray:
    rap_bgr = rap_rgb[:, :, ::-1]
    target_h = bev_bgr.shape[0]
    rap_w = int(round(rap_bgr.shape[1] * (target_h / max(rap_bgr.shape[0], 1))))
    rap_resized = cv2.resize(rap_bgr, (rap_w, target_h), interpolation=cv2.INTER_AREA)
    cv2.putText(rap_resized, "RAP camera render", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    strip = cv2.hconcat([bev_bgr, rap_resized])
    bar_h = 42
    out = np.zeros((strip.shape[0] + bar_h, strip.shape[1], 3), dtype=np.uint8)
    out[bar_h:, :, :] = strip
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = args.out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.csv"

    env = Drive(
        num_agents=args.num_agents,
        num_maps=args.num_maps,
        map_dir=args.map_dir,
        control_mode=args.control_mode,
        init_mode=args.init_mode,
        episode_length=args.episode_length,
        resample_frequency=0,
        report_interval=max(args.frames, 1),
    )

    try:
        env.reset(seed=args.seed)
        state0 = env.get_global_agent_state()
        ego_idx = choose_ego_index(valid_agent_indices(state0), args.ego_agent_index, args.ego_select_mode)
        ego_id = int(state0["id"][ego_idx])

        replay_states: List[Dict[str, np.ndarray]] | None = None
        gt_heading_lookup: Dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        if args.replay_captured:
            if args.replay_source == "ground_truth":
                replay_states = capture_ground_truth_rollout_states(env, args.frames)
            else:
                replay_states = capture_rollout_states(env, args.frames)
                gt_heading_lookup = build_ground_truth_heading_lookup(env, args.frames, state_ref=replay_states[0])
            state_ref = replay_states[0]
        else:
            state_ref = state0

        road_edges = env.get_road_edge_polylines()
        sid, boundary_polylines_abs, _, sid_candidates = choose_scenario_id_for_ego_from_road_edges(
            road_edges=road_edges,
            ego_x=float(state_ref["x"][ego_idx]),
            ego_y=float(state_ref["y"][ego_idx]),
            map_radius=args.map_radius,
        )
        if not boundary_polylines_abs:
            raise RuntimeError("No road-edge polylines available for selected scenario")

        renderer = ScenarioRenderer(
            camera_channel_list=args.cameras,
            width=args.render_width,
            height=args.render_height,
        )
        maybe_rescale_renderer_intrinsics(
            renderer=renderer,
            width=args.render_width,
            height=args.render_height,
            enabled=(not args.no_rescale_intrinsics),
        )

        print(f"Output directory: {args.out_dir}")
        print(f"Scenario id: {sid}")
        print(f"Scenario id candidates in road edges: {sid_candidates}")
        print(f"Ego index: {ego_idx}")
        print(f"Ego id: {ego_id}")
        print(f"Ego select mode: {args.ego_select_mode}")
        print(f"Cameras: {args.cameras}")
        print(f"Panel camera: {args.panel_camera}")
        if args.replay_captured:
            print(f"Replay mode: {args.replay_source}")
        else:
            print("Replay mode: step_neutral_actions")
        if args.show_native_window and args.replay_captured:
            print("[warn] --show-native-window is less meaningful with --replay-captured")

        fieldnames = [
            "frame",
            "scenario_id",
            "ego_x",
            "ego_y",
            "raw_heading",
            "camera_yaw",
            "map_features",
            "agent_boxes",
        ] + [f"nonzero_{cam}" for cam in args.cameras]
        rows: List[Dict[str, object]] = []

        current_ego_idx = resolve_ego_index_by_id(state_ref, ego_id=ego_id, preferred_fallback=ego_idx)
        last_ego_x = float(state_ref["x"][current_ego_idx])
        last_ego_y = float(state_ref["y"][current_ego_idx])
        last_ego_heading = float(state_ref["heading"][current_ego_idx])

        map_bin_path = infer_map_bin_path(env, args.map_dir)
        lane_polylines_abs: List[np.ndarray] = []
        if map_bin_path is not None:
            try:
                lane_polylines_abs = extract_lane_polylines_abs_from_map_bin(map_bin_path)
            except Exception as e:
                print(f"[warn] failed to parse lane polylines from {map_bin_path}: {e}")
        print(f"Lane source polylines: {len(lane_polylines_abs)}")
        static_clip_bound = max(args.map_radius * 5.0, 200.0)
        static_map_features_preview = build_boundary_map_features_static(
            boundary_polylines_abs,
            lane_polylines_abs=lane_polylines_abs,
            clip_bound=static_clip_bound,
            ego_x=float(state_ref["x"][current_ego_idx]),
            ego_y=float(state_ref["y"][current_ego_idx]),
        )
        print(f"Static boundary features: {len(static_map_features_preview)}")

        for t in range(args.frames):
            if args.replay_captured:
                assert replay_states is not None
                state = replay_states[t]
                apply_ground_truth_headings_to_state(state, t, gt_heading_lookup)
            else:
                state = env.get_global_agent_state()
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
                ego_x, ego_y, raw_heading = last_ego_x, last_ego_y, last_ego_heading
                state["x"][current_ego_idx] = ego_x
                state["y"][current_ego_idx] = ego_y
                state["heading"][current_ego_idx] = raw_heading

            camera_yaw = compute_camera_yaw(raw_heading, args)

            map_features = build_boundary_map_features_static(
                boundary_polylines_abs,
                lane_polylines_abs=lane_polylines_abs,
                clip_bound=static_clip_bound,
                ego_x=ego_x,
                ego_y=ego_y,
            )
            scenario = make_scenario(
                state=state,
                map_features=map_features,
                ego_idx=current_ego_idx,
                include_ego_box=args.include_ego_box,
                assumed_height=args.assumed_height,
                agent_radius=args.agent_radius,
                ego_heading=camera_yaw,
            )
            rendered = renderer.observe(scenario)
            counts = get_nonzero_counts(rendered)

            bev = draw_bev_panel(
                scenario=scenario,
                size=args.bev_size,
                radius=args.map_radius,
                camera_yaw=camera_yaw,
            )
            label = (
                f"frame={t:04d} sid={sid} map_features={len(map_features)} boxes={scenario['anns']['gt_boxes_world'].shape[0]} "
                f"heading(raw/cam)=({raw_heading:.3f}/{camera_yaw:.3f}) nonzero_{args.panel_camera}={counts.get(args.panel_camera, 0)}"
            )
            panel = compose_panel(bev, rendered[args.panel_camera], label)
            panel_path = panel_dir / f"frame_{t:04d}_compare.jpg"
            cv2.imwrite(str(panel_path), panel)

            row: Dict[str, object] = {
                "frame": t,
                "scenario_id": sid,
                "ego_x": f"{ego_x:.5f}",
                "ego_y": f"{ego_y:.5f}",
                "raw_heading": f"{raw_heading:.6f}",
                "camera_yaw": f"{camera_yaw:.6f}",
                "map_features": len(map_features),
                "agent_boxes": int(scenario["anns"]["gt_boxes_world"].shape[0]),
            }
            for cam in args.cameras:
                row[f"nonzero_{cam}"] = int(counts.get(cam, 0))
            rows.append(row)

            if args.show_native_window and (t % args.native_render_every == 0):
                env.render()

            if not args.replay_captured:
                step_env_with_zeros(env)

        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[done] wrote panels: {panel_dir}")
        print(f"[done] wrote metrics: {metrics_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
