#!/usr/bin/env python3
"""Export CAM_FRONT keyframes from a nuScenes scene into an MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CAM_FRONT keyframes as an MP4 video."
    )
    parser.add_argument("--dataroot", type=Path, default=Path("/data/nuscenes"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--scene-id", default="0039", help="Scene suffix, e.g. 0039")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .mp4 path (default: outputs/scene-<id>_cam_front.mp4)",
    )
    parser.add_argument("--fps", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_name = f"scene-{args.scene_id}"
    output_path = (
        args.output
        if args.output is not None
        else Path("outputs") / f"{scene_name}_cam_front.mp4"
    )

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    if scene is None:
        raise ValueError(f"{scene_name} not found in {args.version}")

    frame_paths: list[Path] = []
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        cam_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        frame_paths.append(args.dataroot / cam_sd["filename"])
        sample_token = sample["next"]

    if not frame_paths:
        raise RuntimeError(f"No CAM_FRONT frames found for {scene_name}")

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Failed to read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    written = 0
    for frame_path in tqdm(frame_paths, desc=f"Encoding {scene_name}", unit="frame"):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1

    writer.release()
    print(f"Scene: {scene_name}")
    print(f"Frames discovered: {len(frame_paths)}")
    print(f"Frames written: {written}")
    print(f"Video path: {output_path.resolve()}")


if __name__ == "__main__":
    main()
