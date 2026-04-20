#!/usr/bin/env python3
"""Colorize one or more keyframes of a nuScenes scene and write point cloud outputs.

Default: `.pcd` (Open3D). Use `--format ply` for PLY.

Default: scene-0039, first keyframe, all 6 cameras, ego-frame output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colorization import (  # noqa: E402
    CAMERAS_360,
    ColorizationConfig,
    colorize_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nuScenes multi-camera colorization")
    parser.add_argument("--dataroot", type=Path, default=Path("/data/nuscenes"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--scene-id", default="0039", help="Scene suffix, e.g. 0039")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Keyframe index within scene (0=first). Use -1 for ALL keyframes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/colorized"),
    )
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--border-margin", type=int, default=4)
    parser.add_argument(
        "--occlusion-bucket",
        type=int,
        default=1,
        help="Pixel bucket for image-space z-buffer. 1=strict, >=2 coarser.",
    )
    parser.add_argument(
        "--n-sweeps",
        type=int,
        default=1,
        help="Number of LiDAR sweeps to aggregate per keyframe (>=1).",
    )
    parser.add_argument(
        "--output-frame",
        default="ego_lidar",
        choices=["ego_lidar", "lidar"],
    )
    parser.add_argument(
        "--filter-dynamic-points",
        action="store_true",
        help=(
            "Remove points that fall inside dynamic-object boxes "
            "(vehicle/human/animal) at the LiDAR keyframe timestamp."
        ),
    )
    parser.add_argument(
        "--dynamic-category-prefix",
        action="append",
        default=None,
        help=(
            "Category prefix considered dynamic. Repeat to add multiple, "
            "e.g. --dynamic-category-prefix vehicle. "
            "--dynamic-category-prefix human."
        ),
    )
    parser.add_argument(
        "--format",
        default="pcd",
        choices=["pcd", "ply"],
        help="Point cloud file format (default: pcd).",
    )
    return parser.parse_args()


def collect_sample_tokens(nusc: NuScenes, scene_name: str, index: int) -> list[str]:
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    if scene is None:
        raise ValueError(f"Scene {scene_name} not found")

    tokens: list[str] = []
    t = scene["first_sample_token"]
    while t:
        tokens.append(t)
        t = nusc.get("sample", t)["next"]

    if index == -1:
        return tokens
    if index < 0 or index >= len(tokens):
        raise IndexError(
            f"sample-index {index} out of range; scene has {len(tokens)} keyframes"
        )
    return [tokens[index]]


def write_point_cloud(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def main() -> None:
    args = parse_args()
    scene_name = f"scene-{args.scene_id}"
    cfg = ColorizationConfig(
        min_depth=args.min_depth,
        border_margin_px=args.border_margin,
        image_pixel_bucket=args.occlusion_bucket,
        n_sweeps=args.n_sweeps,
        output_frame=args.output_frame,
        filter_dynamic_points=args.filter_dynamic_points,
        dynamic_category_prefixes=tuple(args.dynamic_category_prefix)
        if args.dynamic_category_prefix
        else ColorizationConfig.dynamic_category_prefixes,
    )

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    sample_tokens = collect_sample_tokens(nusc, scene_name, args.sample_index)

    output_dir = args.output_dir / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene: {scene_name}")
    print(f"Cameras: {', '.join(CAMERAS_360)}")
    print(f"Keyframes to process: {len(sample_tokens)}")
    print(f"LiDAR sweeps per keyframe: {cfg.n_sweeps}")
    print(f"Dynamic point filtering: {cfg.filter_dynamic_points}")
    if cfg.filter_dynamic_points:
        print(f"Dynamic category prefixes: {', '.join(cfg.dynamic_category_prefixes)}")
    print(f"Output frame: {cfg.output_frame}")
    print(f"Output format: {args.format}")

    ext = args.format
    sweep_suffix = f"_sweeps_{cfg.n_sweeps}"
    summary = []
    for i, token in enumerate(tqdm(sample_tokens, desc="Colorizing", unit="frame")):
        xyz, rgb, cam_idx = colorize_sample(nusc, token, CAMERAS_360, cfg)
        out_path = output_dir / f"{scene_name}_kf{i:03d}_{token[:8]}{sweep_suffix}.{ext}"
        write_point_cloud(out_path, xyz, rgb)
        per_cam = {CAMERAS_360[c]: int(np.sum(cam_idx == c)) for c in range(len(CAMERAS_360))}
        summary.append((out_path, xyz.shape[0], per_cam))

    print("\nDone.")
    for out_path, n_pts, per_cam in summary:
        print(f"- {out_path} | points={n_pts} | per_cam={per_cam}")


if __name__ == "__main__":
    main()
