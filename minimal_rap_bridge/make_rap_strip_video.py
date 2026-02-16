#!/usr/bin/env python3
"""Build a horizontal multi-camera RAP video from rendered frame images."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


FRAME_RE = re.compile(r"^frame_(\d+)_(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stitched RAP strip video (e.g., CAM_L0|CAM_F0|CAM_R0)")
    parser.add_argument("--rap-dir", type=Path, required=True, help="Directory containing frame_XXXX_<CAM>.jpg")
    parser.add_argument("--out-video", type=Path, required=True, help="Output video path (.mp4/.avi)")
    parser.add_argument("--cameras", type=str, default="CAM_L0,CAM_F0,CAM_R0")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "ffmpeg", "opencv"],
        help="auto prefers ffmpeg when available, else OpenCV writer",
    )
    parser.add_argument(
        "--opencv-codec",
        type=str,
        default="MJPG",
        help="OpenCV FOURCC for backend=opencv (e.g., MJPG, XVID, mp4v)",
    )
    parser.add_argument("--label", action="store_true", help="Draw camera labels at top-left of each pane")
    return parser.parse_args()


def parse_cameras(spec: str) -> List[str]:
    cameras = [camera.strip() for camera in spec.split(",") if camera.strip()]
    if not cameras:
        raise ValueError("No cameras provided")
    return cameras


def frame_index_from_path(path: Path) -> int | None:
    match = FRAME_RE.match(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def build_camera_index(rap_dir: Path, camera: str) -> Dict[int, Path]:
    index: Dict[int, Path] = {}
    for path in sorted(rap_dir.glob(f"frame_*_{camera}.jpg")):
        frame_idx = frame_index_from_path(path)
        if frame_idx is None:
            continue
        index[frame_idx] = path
    return index


def load_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    return frame


def resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    current_h, current_w = image.shape[:2]
    if current_h == target_h:
        return image
    target_w = int(round(current_w * (target_h / max(current_h, 1))))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def compose_strip(frames: List[np.ndarray], labels: List[str], draw_labels: bool) -> np.ndarray:
    target_h = min(frame.shape[0] for frame in frames)
    panes: List[np.ndarray] = []
    for frame, label in zip(frames, labels):
        pane = resize_to_height(frame, target_h)
        if draw_labels:
            cv2.putText(pane, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        panes.append(pane)
    return cv2.hconcat(panes)


def write_video_with_opencv(strips: List[np.ndarray], out_video: Path, fps: int, codec: str) -> None:
    if not strips:
        raise RuntimeError("No strip frames to encode")
    if len(codec) != 4:
        raise ValueError("--opencv-codec must be a 4-character FOURCC")

    h, w = strips[0].shape[:2]
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open OpenCV writer for {out_video} with codec={codec}")
    try:
        for strip in strips:
            if strip.shape[:2] != (h, w):
                strip = cv2.resize(strip, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(strip)
    finally:
        writer.release()


def write_video_with_ffmpeg(strips: List[np.ndarray], out_video: Path, fps: int) -> None:
    if not strips:
        raise RuntimeError("No strip frames to encode")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    with tempfile.TemporaryDirectory(prefix="rap_strip_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx, strip in enumerate(strips):
            frame_path = tmp_path / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), strip)

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_path / "frame_%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(out_video),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def main() -> None:
    args = parse_args()
    if not args.rap_dir.exists():
        raise FileNotFoundError(f"RAP dir not found: {args.rap_dir}")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    cameras = parse_cameras(args.cameras)
    per_camera = [build_camera_index(args.rap_dir, camera) for camera in cameras]
    if any(len(index) == 0 for index in per_camera):
        missing = [camera for camera, index in zip(cameras, per_camera) if len(index) == 0]
        raise RuntimeError(f"No frames found for cameras: {missing}")

    frame_ids = sorted(set.intersection(*(set(index.keys()) for index in per_camera)))
    if not frame_ids:
        raise RuntimeError("No common frame indices across requested cameras")

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    strips: List[np.ndarray] = []
    for frame_id in frame_ids:
        frames = [load_frame(index[frame_id]) for index in per_camera]
        strips.append(compose_strip(frames, cameras, args.label))

    if not strips:
        raise RuntimeError("No frames written to strip video")

    backend = args.backend
    if backend == "auto":
        backend = "ffmpeg" if shutil.which("ffmpeg") is not None else "opencv"

    if backend == "ffmpeg":
        try:
            write_video_with_ffmpeg(strips, args.out_video, args.fps)
        except RuntimeError as exc:
            if args.backend == "ffmpeg":
                raise
            print(f"[warn] {exc}")
            print("[warn] Falling back to OpenCV writer")
            backend = "opencv"

    if backend == "opencv":
        write_video_with_opencv(strips, args.out_video, args.fps, args.opencv_codec)

    print(f"[done] strip video: {args.out_video}")
    print(
        f"[done] frames: {len(strips)}, cameras: {','.join(cameras)}, fps={args.fps}, backend={backend}"
    )


if __name__ == "__main__":
    main()
