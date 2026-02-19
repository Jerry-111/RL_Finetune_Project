#!/usr/bin/env python3
"""Run multiple bridge scenes and build quick visual + QA summary."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np


FRAME_RE = re.compile(r"\[frame\s+(\d+)\]\s+nonzero_pixels:\s*(.*)")
COUNT_RE = re.compile(r"([A-Z0-9_]+)=(\d+)")
SCENARIO_RE = re.compile(r"^Scenario id:\s*(.*)$")


@dataclass
class SweepConfig:
    out_root: Path
    seeds: List[int]
    frames: int
    episode_length: int
    map_dir: str
    num_maps: int
    num_agents: int
    cameras: List[str]
    map_radius: float
    ego_agent_index: int
    include_ego_box: bool
    render_width: int
    render_height: int
    rescale_intrinsics: bool
    replay_mode: bool
    camera_yaw_mode: str
    camera_yaw_alpha: float
    camera_yaw_max_step_deg: float
    camera_yaw_fixed_rad: float
    control_mode: str
    init_mode: str


@dataclass
class SceneResult:
    seed: int
    scenario_id: str
    out_dir: Path
    log_path: Path
    best_frame: int
    any_nonzero_ratio: float
    per_cam_ratio: Dict[str, float]


def parse_args() -> SweepConfig:
    parser = argparse.ArgumentParser(description="Sweep multiple seeds and build FOV overview")
    parser.add_argument("--out-root", type=Path, default=Path("/tmp/pd_rap_scene_sweep"))
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--map-dir", type=str, default="resources/drive/binaries")
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=32)
    parser.add_argument("--cameras", type=str, default="CAM_F0,CAM_L0,CAM_R0")
    parser.add_argument("--map-radius", type=float, default=100.0)
    parser.add_argument("--ego-agent-index", type=int, default=6)
    parser.add_argument("--include-ego-box", action="store_true")
    parser.add_argument("--render-width", type=int, default=1920)
    parser.add_argument("--render-height", type=int, default=1120)
    parser.add_argument("--no-rescale-intrinsics", action="store_true")
    parser.add_argument("--replay-captured", action="store_true")
    parser.add_argument(
        "--camera-yaw-mode",
        type=str,
        default="heading",
        choices=["heading", "smoothed", "fixed"],
    )
    parser.add_argument("--camera-yaw-alpha", type=float, default=0.9)
    parser.add_argument("--camera-yaw-max-step-deg", type=float, default=0.0)
    parser.add_argument("--camera-yaw-fixed-rad", type=float, default=0.0)
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
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not seeds:
        raise ValueError("No seeds provided in --seeds")
    if not cameras:
        raise ValueError("No cameras provided in --cameras")
    if args.episode_length < args.frames:
        raise ValueError("--episode-length must be >= --frames")

    return SweepConfig(
        out_root=args.out_root,
        seeds=seeds,
        frames=args.frames,
        episode_length=args.episode_length,
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        num_agents=args.num_agents,
        cameras=cameras,
        map_radius=args.map_radius,
        ego_agent_index=args.ego_agent_index,
        include_ego_box=args.include_ego_box,
        render_width=args.render_width,
        render_height=args.render_height,
        rescale_intrinsics=(not args.no_rescale_intrinsics),
        replay_mode=args.replay_captured,
        camera_yaw_mode=args.camera_yaw_mode,
        camera_yaw_alpha=args.camera_yaw_alpha,
        camera_yaw_max_step_deg=args.camera_yaw_max_step_deg,
        camera_yaw_fixed_rad=args.camera_yaw_fixed_rad,
        control_mode=args.control_mode,
        init_mode=args.init_mode,
    )


def parse_frame_counts(stdout: str) -> Dict[int, Dict[str, int]]:
    per_frame: Dict[int, Dict[str, int]] = {}
    for line in stdout.splitlines():
        m = FRAME_RE.search(line)
        if not m:
            continue
        frame_idx = int(m.group(1))
        counts_blob = m.group(2)
        counts = {k: int(v) for k, v in COUNT_RE.findall(counts_blob)}
        per_frame[frame_idx] = counts
    return per_frame


def parse_scenario_id(stdout: str) -> str:
    for line in stdout.splitlines():
        m = SCENARIO_RE.match(line.strip())
        if m:
            sid = m.group(1).strip()
            return sid if sid else "unknown"
    return "unknown"


def ratio(numer: int, denom: int) -> float:
    return float(numer) / float(max(denom, 1))


def summarize_counts(
    per_frame: Dict[int, Dict[str, int]],
    cameras: Sequence[str],
    expected_frames: int,
) -> tuple[int, float, Dict[str, float]]:
    if not per_frame:
        return 0, 0.0, {cam: 0.0 for cam in cameras}

    best_frame = 0
    best_score = -1
    any_nonzero = 0
    cam_nonzero = {cam: 0 for cam in cameras}

    for frame_idx in sorted(per_frame.keys()):
        counts = per_frame[frame_idx]
        score = 0
        any_pos = False
        for cam in cameras:
            value = int(counts.get(cam, 0))
            score += value
            if value > 0:
                cam_nonzero[cam] += 1
                any_pos = True
        if any_pos:
            any_nonzero += 1
        if score > best_score:
            best_score = score
            best_frame = frame_idx

    per_cam_ratio = {cam: ratio(cam_nonzero[cam], expected_frames) for cam in cameras}
    return best_frame, ratio(any_nonzero, expected_frames), per_cam_ratio


def build_strip_image(
    out_dir: Path,
    frame_idx: int,
    cameras: Sequence[str],
    label: str,
) -> np.ndarray:
    images: List[np.ndarray] = []
    for cam in cameras:
        img_path = out_dir / f"frame_{frame_idx:04d}_{cam}.jpg"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(img, f"missing {cam}", (16, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            img = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(img, cam, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        images.append(img)

    strip = cv2.hconcat(images)
    cv2.putText(strip, label, (12, strip.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return strip


def run_single_scene(cfg: SweepConfig, seed: int) -> SceneResult:
    out_dir = cfg.out_root / f"seed_{seed:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    cmd = [
        sys.executable,
        "minimal_rap_bridge/render_pufferdrive_to_rap.py",
        "--out-dir",
        str(out_dir),
        "--frames",
        str(cfg.frames),
        "--episode-length",
        str(cfg.episode_length),
        "--seed",
        str(seed),
        "--map-dir",
        cfg.map_dir,
        "--num-maps",
        str(cfg.num_maps),
        "--num-agents",
        str(cfg.num_agents),
        "--cameras",
        ",".join(cfg.cameras),
        "--map-radius",
        str(cfg.map_radius),
        "--ego-agent-index",
        str(cfg.ego_agent_index),
        "--render-width",
        str(cfg.render_width),
        "--render-height",
        str(cfg.render_height),
        "--camera-yaw-mode",
        cfg.camera_yaw_mode,
        "--camera-yaw-alpha",
        str(cfg.camera_yaw_alpha),
        "--camera-yaw-max-step-deg",
        str(cfg.camera_yaw_max_step_deg),
        "--camera-yaw-fixed-rad",
        str(cfg.camera_yaw_fixed_rad),
        "--control-mode",
        cfg.control_mode,
        "--init-mode",
        cfg.init_mode,
    ]
    if cfg.include_ego_box:
        cmd.append("--include-ego-box")
    if not cfg.rescale_intrinsics:
        cmd.append("--no-rescale-intrinsics")
    if cfg.replay_mode:
        cmd.append("--replay-captured")

    print(f"[run] seed={seed} out={out_dir}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"Scene run failed for seed={seed}. See {log_path}")

    per_frame = parse_frame_counts(proc.stdout)
    scenario_id = parse_scenario_id(proc.stdout)
    best_frame, any_ratio, per_cam = summarize_counts(
        per_frame=per_frame,
        cameras=cfg.cameras,
        expected_frames=cfg.frames,
    )
    return SceneResult(
        seed=seed,
        scenario_id=scenario_id,
        out_dir=out_dir,
        log_path=log_path,
        best_frame=best_frame,
        any_nonzero_ratio=any_ratio,
        per_cam_ratio=per_cam,
    )


def write_summary_csv(path: Path, results: Sequence[SceneResult], cameras: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["seed", "scenario_id", "out_dir", "best_frame", "any_nonzero_ratio"] + [
            f"{cam}_nonzero_ratio" for cam in cameras
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "seed": r.seed,
                "scenario_id": r.scenario_id,
                "out_dir": str(r.out_dir),
                "best_frame": r.best_frame,
                "any_nonzero_ratio": f"{r.any_nonzero_ratio:.4f}",
            }
            for cam in cameras:
                row[f"{cam}_nonzero_ratio"] = f"{r.per_cam_ratio.get(cam, 0.0):.4f}"
            writer.writerow(row)


def write_overview_image(path: Path, strips: Sequence[np.ndarray]) -> None:
    if not strips:
        return
    gap = 8
    width = max(s.shape[1] for s in strips)
    height = sum(s.shape[0] for s in strips) + gap * (len(strips) - 1)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = 0
    for i, strip in enumerate(strips):
        h, w = strip.shape[:2]
        canvas[y : y + h, :w] = strip
        y += h
        if i < len(strips) - 1:
            y += gap
    cv2.imwrite(str(path), canvas)


def main() -> None:
    cfg = parse_args()
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    results: List[SceneResult] = []
    strips: List[np.ndarray] = []
    for seed in cfg.seeds:
        result = run_single_scene(cfg, seed)
        results.append(result)
        label = (
            f"seed={result.seed} sid={result.scenario_id} "
            f"best_frame={result.best_frame} any_nonzero={result.any_nonzero_ratio:.1%}"
        )
        strips.append(build_strip_image(result.out_dir, result.best_frame, cfg.cameras, label))

    summary_csv = cfg.out_root / "summary.csv"
    write_summary_csv(summary_csv, results, cfg.cameras)

    overview_jpg = cfg.out_root / "overview.jpg"
    write_overview_image(overview_jpg, strips)

    print("\n[done] scene sweep complete")
    print(f"summary: {summary_csv}")
    print(f"overview: {overview_jpg}")


if __name__ == "__main__":
    main()
