#!/usr/bin/env python3
"""Compare PufferDrive native visualizer video against RAP rendered frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build side-by-side native-vs-RAP comparison panels")
    parser.add_argument("--native-video", type=Path, required=True, help="Path to visualize output .mp4")
    parser.add_argument("--rap-dir", type=Path, required=True, help="Directory with frame_XXXX_<CAM>.jpg")
    parser.add_argument("--camera", type=str, default="CAM_F0")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/pd_rap_native_compare"))
    parser.add_argument("--frame-offset", type=int, default=0, help="native_frame = rap_frame + offset")
    parser.add_argument("--sample-every", type=int, default=5, help="Use every Nth aligned frame")
    parser.add_argument("--max-samples", type=int, default=24, help="Max side-by-side pairs to save")
    parser.add_argument("--panel-height", type=int, default=360)
    return parser.parse_args()


def list_rap_frames(rap_dir: Path, camera: str) -> List[Path]:
    return sorted(rap_dir.glob(f"frame_*_{camera}.jpg"))


def parse_rap_frame_idx(path: Path) -> int:
    # frame_0007_CAM_F0.jpg
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return -1
    return int(parts[1])


def read_native_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def resized_with_label(img_bgr: np.ndarray, target_h: int, label: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    target_w = int(round(w * (target_h / max(h, 1))))
    resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    cv2.putText(resized, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
    return resized


def make_pair(native_bgr: np.ndarray, rap_bgr: np.ndarray, meta: str, panel_h: int) -> np.ndarray:
    native_r = resized_with_label(native_bgr, panel_h, "PufferDrive native visualize")
    rap_r = resized_with_label(rap_bgr, panel_h, "RAP bridge render")
    pair = cv2.hconcat([native_r, rap_r])
    bar_h = 36
    out = np.zeros((pair.shape[0] + bar_h, pair.shape[1], 3), dtype=np.uint8)
    out[bar_h:, :, :] = pair
    cv2.putText(out, meta, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (230, 230, 230), 2)
    return out


def write_overview(path: Path, panels: List[np.ndarray]) -> None:
    if not panels:
        return
    gap = 6
    width = max(p.shape[1] for p in panels)
    height = sum(p.shape[0] for p in panels) + gap * (len(panels) - 1)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = 0
    for i, p in enumerate(panels):
        h, w = p.shape[:2]
        canvas[y : y + h, :w] = p
        y += h
        if i < len(panels) - 1:
            y += gap
    cv2.imwrite(str(path), canvas)


def main() -> None:
    args = parse_args()
    if args.sample_every <= 0:
        raise ValueError("--sample-every must be > 0")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")
    if args.panel_height <= 0:
        raise ValueError("--panel-height must be > 0")

    if not args.native_video.exists():
        raise FileNotFoundError(f"Native video not found: {args.native_video}")
    rap_frames = list_rap_frames(args.rap_dir, args.camera)
    if not rap_frames:
        raise RuntimeError(f"No RAP frames found in {args.rap_dir} for camera {args.camera}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = args.out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.out_dir / "metrics.csv"
    overview_jpg = args.out_dir / "overview.jpg"

    cap = cv2.VideoCapture(str(args.native_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open native video: {args.native_video}")
    native_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows: List[dict] = []
    overview_panels: List[np.ndarray] = []
    sample_id = 0
    for rap_path in rap_frames:
        rap_idx = parse_rap_frame_idx(rap_path)
        if rap_idx < 0:
            continue
        if rap_idx % args.sample_every != 0:
            continue
        native_idx = rap_idx + args.frame_offset
        if native_idx < 0 or native_idx >= native_count:
            continue

        native = read_native_frame(cap, native_idx)
        rap = cv2.imread(str(rap_path), cv2.IMREAD_COLOR)
        if native is None or rap is None:
            continue

        meta = f"rap_frame={rap_idx} native_frame={native_idx} offset={args.frame_offset}"
        panel = make_pair(native, rap, meta, args.panel_height)
        panel_path = pairs_dir / f"pair_{sample_id:04d}.jpg"
        cv2.imwrite(str(panel_path), panel)
        overview_panels.append(panel)

        rows.append(
            {
                "sample_id": sample_id,
                "rap_frame": rap_idx,
                "native_frame": native_idx,
                "native_mean_b": f"{float(native[:, :, 0].mean()):.3f}",
                "native_mean_g": f"{float(native[:, :, 1].mean()):.3f}",
                "native_mean_r": f"{float(native[:, :, 2].mean()):.3f}",
                "rap_mean_b": f"{float(rap[:, :, 0].mean()):.3f}",
                "rap_mean_g": f"{float(rap[:, :, 1].mean()):.3f}",
                "rap_mean_r": f"{float(rap[:, :, 2].mean()):.3f}",
            }
        )
        sample_id += 1
        if sample_id >= args.max_samples:
            break

    cap.release()

    if not rows:
        raise RuntimeError("No aligned samples found. Try changing --frame-offset or --sample-every.")

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_overview(overview_jpg, overview_panels)
    print(f"[done] pairs dir: {pairs_dir}")
    print(f"[done] overview: {overview_jpg}")
    print(f"[done] metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
