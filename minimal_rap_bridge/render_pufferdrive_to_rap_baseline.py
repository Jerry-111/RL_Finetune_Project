#!/usr/bin/env python3
"""Trimmed baseline RAP bridge from the known-good no-flip captured run."""

from __future__ import annotations

import argparse
from pathlib import Path

from render_pufferdrive_to_rap import BridgeConfig, run_bridge


def parse_args() -> BridgeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Render RAP frames using the known-good baseline settings: "
            "captured replay + heading camera yaw + no lateral flip."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/pd_rap_baseline"))
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
        help="Comma-separated camera ids from RAP renderer camera_params",
    )
    parser.add_argument("--render-width", type=int, default=1920)
    parser.add_argument("--render-height", type=int, default=1120)
    parser.add_argument(
        "--ego-select-mode",
        type=str,
        default="index",
        choices=["index", "native_visualize", "random_seeded"],
    )
    parser.add_argument("--ego-agent-index", type=int, default=6)
    parser.add_argument(
        "--ego-random-seed",
        type=int,
        default=None,
        help="Seed used when --ego-select-mode random_seeded (default: --seed)",
    )
    parser.add_argument("--include-ego-box", action="store_true")
    parser.add_argument("--assumed-height", type=float, default=1.6)
    parser.add_argument("--map-radius", type=float, default=120.0)
    parser.add_argument(
        "--agent-radius",
        type=float,
        default=None,
        help="Meters around ego to keep agent boxes (default: no radius filtering)",
    )
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
    parser.add_argument(
        "--camera-yaw-mode",
        type=str,
        default="heading",
        choices=["heading", "fixed"],
    )
    parser.add_argument("--camera-yaw-fixed-rad", type=float, default=0.0)
    parser.add_argument(
        "--control-source",
        type=str,
        default="captured_replay",
        choices=["captured_replay", "native_policy"],
        help="Use known-good captured replay baseline or native closed-loop policy stepping",
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
    parser.add_argument(
        "--show-respawned-agents",
        action="store_true",
        help="Do not hide agents that have respawned (native visualize hides them by default)",
    )
    args = parser.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not cameras:
        raise ValueError("At least one camera must be provided via --cameras")
    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.episode_length < args.frames:
        raise ValueError("--episode-length must be >= --frames")

    use_captured_replay = args.control_source == "captured_replay"

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
        replay_mode=use_captured_replay,
        replay_source="captured",
        ego_select_mode=args.ego_select_mode,
        ego_random_seed=(args.seed if args.ego_random_seed is None else args.ego_random_seed),
        camera_yaw_mode=args.camera_yaw_mode,
        camera_yaw_fixed_rad=args.camera_yaw_fixed_rad,
        flip_lateral_axis=False,
        control_source=("native_policy" if not use_captured_replay else "neutral_actions"),
        policy_path=args.policy_path,
        actions_log_path=args.actions_log,
        hide_respawned_agents=(not args.show_respawned_agents),
        timing_csv_path=None,
        scene_summary_json_path=None,
        profile_render_write_only=False,
    )


def main() -> None:
    cfg = parse_args()
    if cfg.replay_mode:
        print("Baseline mode: replay_source=captured, flip_lateral_axis=False")
    else:
        print(f"Baseline mode: native_policy, flip_lateral_axis=False, policy_path={cfg.policy_path}")
    run_bridge(cfg)


if __name__ == "__main__":
    main()
