#!/usr/bin/env python3
"""Smoke test for nuScenes colorized point-cloud pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nuScenes colorization smoke test")
    parser.add_argument("--dataroot", type=Path, default=Path("/data/nuscenes"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--scenes-config", type=Path, default=Path("config/scenes.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--max-points", type=int, default=30000)
    return parser.parse_args()


def load_scene_config(cfg_path: Path) -> tuple[list[str], list[str]]:
    payload = yaml.safe_load(cfg_path.read_text())
    candidates = payload["candidate_scene_ids"]
    selected = payload["selected_scene_ids"]

    if len(selected) > 5:
        raise ValueError("selected_scene_ids must contain at most 5 scenes")

    unknown = sorted(set(selected) - set(candidates))
    if unknown:
        raise ValueError(f"selected_scene_ids contain unknown IDs: {unknown}")

    return candidates, selected


def scene_id(scene_name: str) -> str:
    return scene_name.split("-")[-1]


def transform_lidar_to_camera(
    nusc: NuScenes, lidar_sd: dict, camera_sd: dict
) -> LidarPointCloud:
    pc = LidarPointCloud.from_file(str(Path(nusc.dataroot) / lidar_sd["filename"]))

    cs_lidar = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pose_lidar = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    cs_cam = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
    pose_cam = nusc.get("ego_pose", camera_sd["ego_pose_token"])

    # Lidar sensor -> ego (lidar timestamp)
    pc.rotate(Quaternion(cs_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_lidar["translation"]))

    # Ego -> global
    pc.rotate(Quaternion(pose_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(pose_lidar["translation"]))

    # Global -> ego (camera timestamp)
    pc.translate(-np.array(pose_cam["translation"]))
    pc.rotate(Quaternion(pose_cam["rotation"]).rotation_matrix.T)

    # Ego -> camera
    pc.translate(-np.array(cs_cam["translation"]))
    pc.rotate(Quaternion(cs_cam["rotation"]).rotation_matrix.T)
    return pc


def main() -> None:
    args = parse_args()
    _, selected = load_scene_config(args.scenes_config)

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    scene_by_id = {scene_id(s["name"]): s for s in nusc.scene}

    missing = [sid for sid in selected if sid not in scene_by_id]
    if missing:
        raise ValueError(f"Selected scenes not found in {args.version}: {missing}")

    print("Validated selected scenes:", ", ".join(selected))
    first_scene = scene_by_id[selected[0]]
    first_sample = nusc.get("sample", first_scene["first_sample_token"])
    lidar_sd = nusc.get("sample_data", first_sample["data"]["LIDAR_TOP"])
    cam_sd = nusc.get("sample_data", first_sample["data"]["CAM_FRONT"])
    cam_calib = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])

    pc_cam = transform_lidar_to_camera(nusc, lidar_sd, cam_sd)
    if args.max_points > 0 and pc_cam.points.shape[1] > args.max_points:
        pc_cam.points = pc_cam.points[:, : args.max_points]

    image_path = Path(args.dataroot) / cam_sd["filename"]
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    depths = pc_cam.points[2, :]
    projected = view_points(
        pc_cam.points[:3, :], np.asarray(cam_calib["camera_intrinsic"]), normalize=True
    )
    uv = projected[:2, :]
    valid = (
        (depths > 1.0)
        & (uv[0, :] >= 0)
        & (uv[0, :] < width)
        & (uv[1, :] >= 0)
        & (uv[1, :] < height)
    )

    if not np.any(valid):
        raise RuntimeError("No valid projected points found for smoke test")

    uv_int = uv[:, valid].astype(np.int32)
    colors = image_rgb[uv_int[1], uv_int[0]] / 255.0
    points = pc_cam.points[:3, valid].T

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{first_scene['name']}_cam_front_smoke.ply"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(output_file), pcd)

    print(f"Scene: {first_scene['name']} ({first_scene['token']})")
    print(f"LIDAR file: {lidar_sd['filename']}")
    print(f"Camera file: {cam_sd['filename']}")
    print(f"Projected valid points: {points.shape[0]}")
    print(f"Wrote smoke output: {output_file}")


if __name__ == "__main__":
    main()
