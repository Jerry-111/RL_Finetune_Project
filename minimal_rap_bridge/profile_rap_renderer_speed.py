#!/usr/bin/env python3
"""Profile RAP renderer throughput over deterministic map IDs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


ALL_CAMERAS = "CAM_F0,CAM_L0,CAM_L1,CAM_L2,CAM_R0,CAM_R1,CAM_R2,CAM_B0"
STAGE_ROOT = Path("/tmp/rap_profile_maps")


def parse_map_ids(spec: str) -> List[int]:
    ids: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError as e:
            raise ValueError(f"Invalid map id '{token}' in --map-ids") from e
    if not ids:
        raise ValueError("No map ids provided")
    return ids


def stage_single_map(map_root: Path, map_id: int) -> tuple[Path, Path]:
    source = (map_root / f"map_{map_id:03d}.bin").resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing map binary: {source}")

    stage_dir = STAGE_ROOT / f"map_{map_id:03d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_map = stage_dir / "map_000.bin"
    if staged_map.exists() or staged_map.is_symlink():
        staged_map.unlink()
    staged_map.symlink_to(source)
    return stage_dir, source


def run_single_map(
    map_id: int,
    map_root: Path,
    out_root: Path,
    frames: int,
    episode_length: int,
    seed: int,
    num_agents: int,
    control_mode: str,
    init_mode: str,
) -> Dict[str, float | int | str]:
    stage_dir, source_path = stage_single_map(map_root, map_id)
    scene_dir = out_root / f"map_{map_id:03d}"
    frames_dir = scene_dir / "frames"
    timing_csv = scene_dir / "timing.csv"
    summary_json = scene_dir / "scene_summary.json"
    log_path = scene_dir / "run.log"
    scene_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable,
        "minimal_rap_bridge/render_pufferdrive_to_rap.py",
        "--out-dir",
        str(frames_dir),
        "--frames",
        str(frames),
        "--episode-length",
        str(episode_length),
        "--seed",
        str(seed),
        "--map-dir",
        str(stage_dir),
        "--num-maps",
        "1",
        "--num-agents",
        str(num_agents),
        "--control-mode",
        control_mode,
        "--init-mode",
        init_mode,
        "--cameras",
        ALL_CAMERAS,
        "--timing-csv",
        str(timing_csv),
        "--scene-summary-json",
        str(summary_json),
        "--profile-render-write-only",
    ]
    print(f"[run] map_id={map_id:03d} src={source_path}")
    # Replay modes can trigger native instability in some environments.
    # For throughput profiling, use deterministic neutral-action stepping.
    run_mode = "neutral_actions"
    proc = subprocess.run(base_cmd, text=True, capture_output=True)

    log_path.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Profiling failed for map_{map_id:03d}. See {log_path}.\n"
            f"stderr (tail): {proc.stderr[-500:]}"
        )
    if not summary_json.exists():
        raise RuntimeError(f"Missing scene summary output for map_{map_id:03d}: {summary_json}")
    with open(summary_json, "r", encoding="utf-8") as f:
        scene = json.load(f)

    scene["map_id"] = map_id
    scene["source_map_path"] = str(source_path)
    scene["staged_map_dir"] = str(stage_dir)
    scene["frames_dir"] = str(frames_dir)
    scene["timing_csv"] = str(timing_csv)
    scene["run_log"] = str(log_path)
    scene["replay_mode_used"] = run_mode
    return scene


def summarize_across_scenes(scene_rows: List[Dict[str, float | int | str]]) -> Dict[str, float | int]:
    frame_counts = np.array([int(r["frames"]) for r in scene_rows], dtype=np.int64)
    total_ms = np.array([float(r["total_ms"]) for r in scene_rows], dtype=np.float64)
    scene_mean_ms = np.array([float(r["mean_frame_ms"]) for r in scene_rows], dtype=np.float64)
    total_frames = int(np.sum(frame_counts))
    weighted_mean = float(np.sum(total_ms) / total_frames) if total_frames > 0 else 0.0
    return {
        "num_scenes": int(len(scene_rows)),
        "total_frames": total_frames,
        "total_ms": float(np.sum(total_ms)),
        "mean_frame_ms_weighted": weighted_mean,
        "p50_scene_mean_ms": float(np.percentile(scene_mean_ms, 50)) if scene_mean_ms.size else 0.0,
        "p95_scene_mean_ms": float(np.percentile(scene_mean_ms, 95)) if scene_mean_ms.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile RAP render speed on fixed map ids")
    parser.add_argument("--map-root", type=Path, default=Path("resources/drive/binaries/validation"))
    parser.add_argument("--map-ids", type=str, default="0,1,2,3,4")
    parser.add_argument("--frames", type=int, default=91)
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--out-root", type=Path, default=Path("/tmp/rap_renderer_profile"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=32)
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
        "--clean-stage-root",
        action="store_true",
        help="Delete /tmp/rap_profile_maps before running",
    )
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.episode_length < args.frames:
        raise ValueError("--episode-length must be >= --frames")

    map_ids = parse_map_ids(args.map_ids)
    args.map_root = args.map_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.clean_stage_root and STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)

    scene_rows: List[Dict[str, float | int | str]] = []
    for map_id in map_ids:
        scene = run_single_map(
            map_id=map_id,
            map_root=args.map_root,
            out_root=args.out_root,
            frames=args.frames,
            episode_length=args.episode_length,
            seed=args.seed,
            num_agents=args.num_agents,
            control_mode=args.control_mode,
            init_mode=args.init_mode,
        )
        scene_rows.append(scene)
        print(
            f"[scene] map={map_id:03d} frames={scene['frames']} total_ms={float(scene['total_ms']):.3f} "
            f"mean_frame_ms={float(scene['mean_frame_ms']):.3f} p95={float(scene['p95_frame_ms']):.3f} "
            f"fps={float(scene['fps_equiv']):.3f}"
        )

    summary = summarize_across_scenes(scene_rows)
    summary_json = args.out_root / "profile_summary.json"
    summary_csv = args.out_root / "profile_summary.csv"

    payload = {
        "config": {
            "map_root": str(args.map_root),
            "map_ids": map_ids,
            "frames": int(args.frames),
            "episode_length": int(args.episode_length),
            "out_root": str(args.out_root),
            "seed": int(args.seed),
            "num_agents": int(args.num_agents),
            "control_mode": args.control_mode,
            "init_mode": args.init_mode,
            "cameras": ALL_CAMERAS.split(","),
            "timing_mode": "render_plus_jpeg_write",
            "replay_mode": "off",
            "control_source": "neutral_actions",
        },
        "scenes": scene_rows,
        "aggregate": summary,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "map_id",
            "frames",
            "total_ms",
            "mean_frame_ms",
            "p50_frame_ms",
            "p95_frame_ms",
            "min_frame_ms",
            "max_frame_ms",
            "fps_equiv",
            "source_map_path",
            "frames_dir",
            "timing_csv",
            "run_log",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print("\n=== Scene Summary ===")
    for row in scene_rows:
        print(
            f"map={int(row['map_id']):03d} frames={int(row['frames'])} "
            f"total_ms={float(row['total_ms']):.3f} mean_frame_ms={float(row['mean_frame_ms']):.3f} "
            f"p95={float(row['p95_frame_ms']):.3f} fps={float(row['fps_equiv']):.3f}"
        )
    print("=== Aggregate ===")
    print(
        f"total_frames={summary['total_frames']} total_ms={float(summary['total_ms']):.3f} "
        f"mean_frame_ms_weighted={float(summary['mean_frame_ms_weighted']):.3f} "
        f"p50_scene_mean_ms={float(summary['p50_scene_mean_ms']):.3f} "
        f"p95_scene_mean_ms={float(summary['p95_scene_mean_ms']):.3f}"
    )
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
