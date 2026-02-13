#!/usr/bin/env python3
"""Generate RAP rasterized frames that are visually closer to project examples.

This script builds deterministic synthetic map features and lane-aligned vehicle boxes,
then renders multi-camera images using ScenarioRenderer.
"""

import argparse
import math
import os
from typing import Dict, List

import cv2
import numpy as np

from helpers.renderer import ScenarioRenderer


def lane_centerline(x_vals: np.ndarray, lane_offset: float, curve_strength: float) -> np.ndarray:
    """Smooth centerline with mild sinusoidal curvature."""
    y_vals = lane_offset + curve_strength * np.sin(0.03 * x_vals + lane_offset * 0.08)
    return np.stack([x_vals, y_vals], axis=1).astype(np.float32)


def lane_tangent_yaw(x: float, lane_offset: float, curve_strength: float) -> float:
    """Heading aligned with local lane tangent."""
    dydx = curve_strength * 0.03 * math.cos(0.03 * x + lane_offset * 0.08)
    return math.atan2(dydx, 1.0)


def build_map_features() -> Dict[str, Dict[str, np.ndarray]]:
    """Construct dense lane/boundary/crosswalk feature set."""
    features: Dict[str, Dict[str, np.ndarray]] = {}
    x_vals = np.linspace(4.0, 130.0, 220, dtype=np.float32)

    # Multiple road lanes
    for idx, lane_offset in enumerate(np.linspace(-21.0, 21.0, 15)):
        poly = lane_centerline(x_vals, float(lane_offset), curve_strength=2.0)
        features[f"lane_{idx:02d}"] = {
            "type": "LANE",
            "polygon": poly,
        }

    # Left/right boundaries
    left_boundary = lane_centerline(x_vals, lane_offset=-27.0, curve_strength=2.4)
    right_boundary = lane_centerline(x_vals, lane_offset=27.0, curve_strength=2.4)
    features["boundary_left"] = {"type": "BOUNDARY", "polyline": left_boundary}
    features["boundary_right"] = {"type": "BOUNDARY", "polyline": right_boundary}

    # A few crosswalks
    for idx, x0 in enumerate([34.0, 65.0, 98.0]):
        crosswalk = np.array(
            [[x0, -4.5], [x0 + 6.0, -4.5], [x0 + 6.0, 4.5], [x0, 4.5], [x0, -4.5]],
            dtype=np.float32,
        )
        features[f"crosswalk_{idx:02d}"] = {"type": "CROSSWALK", "polygon": crosswalk}

    return features


def build_agent_boxes(seed: int, include_side_agents: bool) -> np.ndarray:
    """Create lane-aligned 3D boxes in RAP gt_boxes_world format."""
    rng = np.random.default_rng(seed)
    boxes: List[List[float]] = []

    lane_offsets = list(np.linspace(-18.0, 18.0, 9))
    long_positions = np.linspace(8.0, 118.0, 24)

    # Main traffic flow: mostly forward cars aligned to lane tangent
    for lane_offset in lane_offsets:
        for x in long_positions:
            if rng.random() > 0.55:
                continue
            y = float(lane_offset + 2.0 * math.sin(0.03 * x + lane_offset * 0.08))
            yaw = lane_tangent_yaw(float(x), float(lane_offset), curve_strength=2.0)
            yaw += float(rng.normal(0.0, 0.07))

            length = float(rng.uniform(3.9, 5.3))
            width = float(rng.uniform(1.75, 2.25))
            height = float(rng.uniform(1.45, 1.95))
            boxes.append([float(x), y, 0.0, length, width, height, yaw])

    # Optional side traffic to better populate L/R cameras
    if include_side_agents:
        side_offsets = [-30.0, -36.0, 30.0, 36.0]
        for side in side_offsets:
            for x in np.linspace(12.0, 96.0, 14):
                if rng.random() > 0.45:
                    continue
                y = float(side + 1.2 * math.sin(0.028 * x + side * 0.03))
                yaw = lane_tangent_yaw(float(x), float(side), curve_strength=1.2)
                yaw += float(rng.normal(0.0, 0.09))
                boxes.append(
                    [
                        float(x),
                        y,
                        0.0,
                        float(rng.uniform(3.9, 5.2)),
                        float(rng.uniform(1.7, 2.3)),
                        float(rng.uniform(1.4, 2.0)),
                        float(yaw),
                    ]
                )

    return np.array(boxes, dtype=np.float32)


def build_scenario(ego_heading: float, seed: int, include_side_agents: bool) -> Dict:
    traffic_lights = [
        (100, True, [42.0, 0.0]),
        (101, False, [58.0, 6.0]),
        (102, True, [58.0, -6.0]),
        (103, False, [82.0, 12.0]),
        (104, True, [82.0, -12.0]),
    ]

    boxes = build_agent_boxes(seed=seed, include_side_agents=include_side_agents)
    names = np.full((boxes.shape[0],), "vehicle", dtype=object)

    return {
        "ego_heading": float(ego_heading),
        "traffic_lights": traffic_lights,
        "map_features": build_map_features(),
        "anns": {
            "gt_boxes_world": boxes,
            "gt_names": names,
        },
    }


def render_sequence(
    output_dir: str,
    width: int,
    height: int,
    seed: int,
    include_side_agents: bool,
    headings: List[float],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    renderer = ScenarioRenderer(camera_channel_list=["CAM_F0", "CAM_L0", "CAM_R0"], width=width, height=height)

    for frame_idx, heading in enumerate(headings):
        scenario = build_scenario(ego_heading=heading, seed=seed + frame_idx, include_side_agents=include_side_agents)
        rendered = renderer.observe(scenario)

        for cam_id, image in rendered.items():
            path = os.path.join(output_dir, f"frame_{frame_idx:02d}_{cam_id}.jpg")
            cv2.imwrite(path, image[:, :, ::-1])
            nonzero = int((image > 0).sum())
            print(f"{path} nonzero={nonzero}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render richer synthetic RAP rasterized scenes")
    parser.add_argument("--output-dir", type=str, default="/tmp/rap_example_like", help="Output image directory")
    parser.add_argument("--width", type=int, default=1280, help="Rendered image width")
    parser.add_argument("--height", type=int, default=720, help="Rendered image height")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--headings",
        type=float,
        nargs="+",
        default=[-0.10, 0.00, 0.10],
        help="Ego headings (radians) to render as sequential frames",
    )
    parser.add_argument(
        "--no-side-agents",
        action="store_true",
        help="Disable side-lane agents (L/R camera views may become sparse)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_sequence(
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        seed=args.seed,
        include_side_agents=not args.no_side_agents,
        headings=args.headings,
    )


if __name__ == "__main__":
    main()
