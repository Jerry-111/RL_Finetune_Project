#!/usr/bin/env python3
"""Check whether replaying logged native-policy actions reproduces identical Drive states."""

from __future__ import annotations

import argparse
import csv
import zlib
from pathlib import Path
from typing import Dict

import numpy as np

from render_pufferdrive_to_rap import Drive, choose_ego_index, resolve_ego_index_by_id, valid_agent_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify native-policy state consistency under exact action replay")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--map-dir", type=str, default="resources/drive/binaries")
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=21)
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
        choices=["index", "native_visualize", "random_seeded"],
    )
    parser.add_argument("--ego-agent-index", type=int, default=6)
    parser.add_argument(
        "--ego-random-seed",
        type=int,
        default=None,
        help="Seed used when --ego-select-mode random_seeded (default: --seed)",
    )
    parser.add_argument("--policy-path", type=str, default="resources/drive/puffer_drive_weights.bin")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional per-step comparison CSV")
    parser.add_argument("--out-report", type=Path, default=None, help="Optional summary report text")
    args = parser.parse_args()
    if args.frames <= 1:
        raise ValueError("--frames must be > 1")
    if args.episode_length < args.frames:
        raise ValueError("--episode-length must be >= --frames")
    return args


def state_crc32(state: Dict[str, np.ndarray]) -> int:
    crc = 0
    for key in ("id", "x", "y", "heading", "length", "width"):
        arr = np.ascontiguousarray(state[key])
        crc = zlib.crc32(arr.view(np.uint8), crc)
    return int(crc & 0xFFFFFFFF)


def build_env(args: argparse.Namespace) -> Drive:
    return Drive(
        num_agents=args.num_agents,
        num_maps=args.num_maps,
        map_dir=args.map_dir,
        control_mode=args.control_mode,
        init_mode=args.init_mode,
        episode_length=args.episode_length,
        resample_frequency=0,
        report_interval=max(args.frames, 1),
    )


def main() -> None:
    args = parse_args()

    # Pass 1: collect exact native-policy action tensors and reference next-state checksums.
    env_ref = build_env(args)
    actions_seq: list[np.ndarray] = []
    ref_crc_seq: list[int] = []
    ref_ego_pose_seq: list[tuple[float, float, float]] = []
    ego_id = -1
    try:
        env_ref.reset(seed=args.seed)
        if env_ref.num_envs != 1:
            raise RuntimeError(
                f"Expected single-env run for parity, but got num_envs={env_ref.num_envs}. "
                "Use a single-env agent count (for this map/config, --num-agents 21)."
            )
        env_ref.init_native_policy(args.policy_path)

        state0 = env_ref.get_global_agent_state()
        ego_idx = choose_ego_index(
            valid_agent_indices(state0),
            args.ego_agent_index,
            args.ego_select_mode,
            (args.seed if args.ego_random_seed is None else args.ego_random_seed),
        )
        ego_id = int(state0["id"][ego_idx])
        current_ego_idx = int(ego_idx)
        last_ego_x = float(state0["x"][current_ego_idx])
        last_ego_y = float(state0["y"][current_ego_idx])

        for _ in range(args.frames - 1):
            state_pre = env_ref.get_global_agent_state()
            current_ego_idx = resolve_ego_index_by_id(
                state_pre,
                ego_id=ego_id,
                preferred_fallback=current_ego_idx,
                last_ego_xy=(last_ego_x, last_ego_y),
            )
            if np.isfinite(state_pre["x"][current_ego_idx]) and np.isfinite(state_pre["y"][current_ego_idx]):
                last_ego_x = float(state_pre["x"][current_ego_idx])
                last_ego_y = float(state_pre["y"][current_ego_idx])

            env_ref.step_native_policy()
            action_tensor = np.array(env_ref.actions, copy=True)
            actions_seq.append(action_tensor)

            state_post = env_ref.get_global_agent_state()
            post_ego_idx = resolve_ego_index_by_id(
                state_post,
                ego_id=ego_id,
                preferred_fallback=current_ego_idx,
                last_ego_xy=(last_ego_x, last_ego_y),
            )
            ref_crc_seq.append(state_crc32(state_post))
            ref_ego_pose_seq.append(
                (
                    float(state_post["x"][post_ego_idx]),
                    float(state_post["y"][post_ego_idx]),
                    float(state_post["heading"][post_ego_idx]),
                )
            )
            if np.isfinite(state_post["x"][post_ego_idx]) and np.isfinite(state_post["y"][post_ego_idx]):
                last_ego_x = float(state_post["x"][post_ego_idx])
                last_ego_y = float(state_post["y"][post_ego_idx])
            current_ego_idx = int(post_ego_idx)
    finally:
        env_ref.close()

    # Pass 2: replay exact actions via env.step(actions), compare resulting state checksums.
    env_rep = build_env(args)
    rows: list[dict[str, object]] = []
    mismatch_count = 0
    first_mismatch_row: dict[str, object] | None = None
    try:
        env_rep.reset(seed=args.seed)
        if env_rep.num_envs != 1:
            raise RuntimeError(
                f"Expected single-env run for parity, but got num_envs={env_rep.num_envs}. "
                "Use a single-env agent count (for this map/config, --num-agents 21)."
            )

        state0 = env_rep.get_global_agent_state()
        current_ego_idx = resolve_ego_index_by_id(
            state0,
            ego_id=ego_id,
            preferred_fallback=choose_ego_index(
                valid_agent_indices(state0),
                args.ego_agent_index,
                args.ego_select_mode,
                (args.seed if args.ego_random_seed is None else args.ego_random_seed),
            ),
        )
        last_ego_x = float(state0["x"][current_ego_idx])
        last_ego_y = float(state0["y"][current_ego_idx])

        for t in range(args.frames - 1):
            step_actions = actions_seq[t]
            if step_actions.shape != env_rep.actions.shape:
                raise RuntimeError(
                    f"Action shape mismatch at step {t}: "
                    f"recorded={step_actions.shape}, replay_env={env_rep.actions.shape}"
                )
            env_rep.step(step_actions)

            state_post = env_rep.get_global_agent_state()
            post_ego_idx = resolve_ego_index_by_id(
                state_post,
                ego_id=ego_id,
                preferred_fallback=current_ego_idx,
                last_ego_xy=(last_ego_x, last_ego_y),
            )
            rep_crc = state_crc32(state_post)
            ref_crc = ref_crc_seq[t]
            crc_match = bool(rep_crc == ref_crc)
            if not crc_match:
                mismatch_count += 1

            ref_x, ref_y, ref_h = ref_ego_pose_seq[t]
            rep_x = float(state_post["x"][post_ego_idx])
            rep_y = float(state_post["y"][post_ego_idx])
            rep_h = float(state_post["heading"][post_ego_idx])
            row = {
                "frame": t,
                "ego_slot": int(post_ego_idx),
                "ego_id": int(ego_id),
                "state_crc_ref": int(ref_crc),
                "state_crc_replay": int(rep_crc),
                "state_crc_match": int(crc_match),
                "ego_x_ref": f"{ref_x:.6f}",
                "ego_x_replay": f"{rep_x:.6f}",
                "ego_y_ref": f"{ref_y:.6f}",
                "ego_y_replay": f"{rep_y:.6f}",
                "ego_heading_ref": f"{ref_h:.6f}",
                "ego_heading_replay": f"{rep_h:.6f}",
                "ego_pos_l2_error": f"{float(np.hypot(rep_x - ref_x, rep_y - ref_y)):.9f}",
                "ego_heading_abs_error": f"{abs(rep_h - ref_h):.9f}",
            }
            rows.append(row)
            if (first_mismatch_row is None) and (not crc_match):
                first_mismatch_row = row

            if np.isfinite(rep_x) and np.isfinite(rep_y):
                last_ego_x, last_ego_y = rep_x, rep_y
            current_ego_idx = int(post_ego_idx)
    finally:
        env_rep.close()

    summary_lines = [
        f"frames_checked={args.frames - 1}",
        f"num_mismatch_steps={mismatch_count}",
        f"all_steps_match={mismatch_count == 0}",
        f"ego_id={ego_id}",
    ]
    if first_mismatch_row is None:
        summary_lines.append("first_mismatch=none")
    else:
        summary_lines.append(f"first_mismatch_frame={first_mismatch_row['frame']}")
        summary_lines.append(f"first_mismatch_state_crc_ref={first_mismatch_row['state_crc_ref']}")
        summary_lines.append(f"first_mismatch_state_crc_replay={first_mismatch_row['state_crc_replay']}")
        summary_lines.append(f"first_mismatch_ego_pos_l2_error={first_mismatch_row['ego_pos_l2_error']}")
        summary_lines.append(f"first_mismatch_ego_heading_abs_error={first_mismatch_row['ego_heading_abs_error']}")

    summary = "\n".join(summary_lines)
    print(summary)

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame"])
            writer.writeheader()
            if rows:
                writer.writerows(rows)
        print(f"[done] wrote csv: {args.out_csv}")

    if args.out_report is not None:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(summary + "\n", encoding="utf-8")
        print(f"[done] wrote report: {args.out_report}")


if __name__ == "__main__":
    main()
