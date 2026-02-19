#!/usr/bin/env python3
"""Create per-seed videos from rendered frame sequences."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build videos from rendered seed frame folders")
    parser.add_argument("--sweep-root", type=Path, required=True, help="Path like /tmp/pd_rap_scene_sweep_val_5_clean")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated seed indices")
    parser.add_argument("--camera", type=str, default="CAM_F0")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <sweep-root>/videos")
    parser.add_argument(
        "--format",
        type=str,
        default="mp4",
        choices=["mp4", "avi"],
        help="Container format; use avi for max compatibility if your player fails on mp4",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "ffmpeg", "opencv"],
        help="Video writing backend. auto prefers ffmpeg when available.",
    )
    parser.add_argument(
        "--opencv-codec",
        type=str,
        default="MJPG",
        help="FOURCC used when backend=opencv (e.g., MJPG, XVID, mp4v)",
    )
    parser.add_argument(
        "--no-verify-decode",
        action="store_true",
        help="Skip quick decode verification after writing each video",
    )
    return parser.parse_args()


def parse_seed_list(spec: str) -> List[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def write_video_from_images_opencv(
    image_paths: List[Path], out_path: Path, fps: int, codec: str
) -> None:
    if not image_paths:
        raise RuntimeError(f"No frames for {out_path.name}")
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {image_paths[0]}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path} with codec={codec}")
    for image_path in image_paths:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()


def write_video_from_images_ffmpeg(
    seed_dir: Path,
    camera: str,
    out_path: Path,
    fps: int,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    input_pattern = seed_dir / f"frame_%04d_{camera}.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(input_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path.name}: {proc.stderr.strip()}")


def verify_video_decode(out_path: Path) -> None:
    cap = cv2.VideoCapture(str(out_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Decode check failed for {out_path}; file may be corrupted or unsupported")

    means = frame.mean(axis=(0, 1))
    b, g, r = float(means[0]), float(means[1]), float(means[2])
    if g > 1.7 * max(b, r + 1e-6):
        print(f"[warn] decode check: first frame is strongly green-dominant for {out_path.name}")


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    out_dir = args.out_dir if args.out_dir is not None else (args.sweep_root / "videos")
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_dir = args.sweep_root / f"seed_{seed:04d}"
        frames = sorted(seed_dir.glob(f"frame_*_{args.camera}.jpg"))
        out_path = out_dir / f"seed_{seed:04d}_{args.camera}.{args.format}"

        used_backend = args.backend
        if args.backend == "auto":
            used_backend = "ffmpeg" if shutil.which("ffmpeg") is not None else "opencv"

        if used_backend == "ffmpeg":
            try:
                write_video_from_images_ffmpeg(seed_dir, args.camera, out_path, args.fps)
                if not args.no_verify_decode:
                    verify_video_decode(out_path)
                print(f"[ok] {out_path} ({len(frames)} frames, backend=ffmpeg, codec=libx264)")
                continue
            except RuntimeError as exc:
                print(f"[warn] {exc}")
                print("[warn] Falling back to OpenCV writer")
                used_backend = "opencv"

        write_video_from_images_opencv(frames, out_path, args.fps, args.opencv_codec)
        if not args.no_verify_decode:
            verify_video_decode(out_path)
        print(
            f"[ok] {out_path} ({len(frames)} frames, backend=opencv, codec={args.opencv_codec})"
        )

    print(f"[done] wrote videos to {out_dir}")


if __name__ == "__main__":
    main()
