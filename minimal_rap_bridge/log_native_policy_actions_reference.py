#!/usr/bin/env python3
"""Log native-policy action decisions from a no-render reference loop."""

from __future__ import annotations

import argparse
import csv
import zlib
from pathlib import Path

import numpy as np

from render_pufferdrive_to_rap import Drive, choose_ego_index, resolve_ego_index_by_id, valid_agent_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log native-policy actions from reference Drive loop")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--map-dir", type=str, default="resources/drive/binaries")
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=32)
    parser.add_argument(
        "--control-mode",
        type=str,
        default="control_vehicles",
        choices=["control_vehicles", "control_agents", "control_wosac", "control_sdc_only"],
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="create_all_valid",
        choices=["create_all_valid", "create_only_controlled"],
    )
    parser.add_argument(
        "--ego-select-mode",
        type=str,
        default="native_visualize",
        choices=["index", "native_visualize"],
    )
    parser.add_argument("--ego-agent-index", type=int, default=6)
    parser.add_argument("--policy-path", type=str, default="resources/drive/puffer_drive_weights.bin")
    args = parser.parse_args()
    if args.frames <= 1:
        raise ValueError("--frames must be > 1 for transition logging")
    if args.episode_length < args.frames:
        raise ValueError("--episode-length must be >= --frames")
    return args


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

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
        env.init_native_policy(args.policy_path)
        state0 = env.get_global_agent_state()
        ego_idx = choose_ego_index(valid_agent_indices(state0), args.ego_agent_index, args.ego_select_mode)
        ego_id = int(state0["id"][ego_idx])
        current_ego_idx = int(ego_idx)
        last_ego_x = float(state0["x"][current_ego_idx])
        last_ego_y = float(state0["y"][current_ego_idx])

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
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
            writer.writeheader()

            for t in range(args.frames - 1):
                state = env.get_global_agent_state()
                current_ego_idx = resolve_ego_index_by_id(
                    state,
                    ego_id=ego_id,
                    preferred_fallback=current_ego_idx,
                    last_ego_xy=(last_ego_x, last_ego_y),
                )
                if np.isfinite(state["x"][current_ego_idx]) and np.isfinite(state["y"][current_ego_idx]):
                    last_ego_x = float(state["x"][current_ego_idx])
                    last_ego_y = float(state["y"][current_ego_idx])

                env.step_native_policy()
                acts = np.array(env.actions, dtype=np.int32, copy=False).reshape(-1)
                ego_action = int(acts[current_ego_idx]) if 0 <= current_ego_idx < acts.shape[0] else -1
                writer.writerow(
                    {
                        "frame": t,
                        "ego_slot": int(current_ego_idx),
                        "ego_id": int(ego_id),
                        "ego_action": ego_action,
                        "action_count": int(acts.shape[0]),
                        "action_min": int(np.min(acts)) if acts.size else 0,
                        "action_max": int(np.max(acts)) if acts.size else 0,
                        "action_sum": int(np.sum(acts, dtype=np.int64)) if acts.size else 0,
                        "action_crc32": int(zlib.crc32(acts.tobytes()) & 0xFFFFFFFF),
                    }
                )
        print(f"[done] wrote reference actions log: {args.out_csv}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
