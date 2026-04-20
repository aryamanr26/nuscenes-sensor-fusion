"""Multi-camera LiDAR colorization for nuScenes.

Pipeline (per keyframe):
  1. Load LIDAR_TOP point cloud (sensor frame).
  1b. Optionally remove points that fall inside dynamic-object boxes at the
      LiDAR keyframe timestamp.
  2. For each of 6 cameras, transform points: LIDAR -> ego(t_lidar) -> global
     -> ego(t_cam) -> camera. This handles sensor-pose timing differences.
  3. Project to image plane using camera intrinsics. Keep only points that fall
     within the image and have positive depth.
  4. Per-camera occlusion handling: bucket points by integer pixel and keep
     only the nearest point per pixel (fast z-buffer in image space).
  5. Multi-camera fusion: each surviving point belongs to one camera. If a
     world point projects into multiple cameras, pick the one with the highest
     quality score (smaller incidence angle, closer to image center, smaller
     depth) to avoid seams.
  6. Sample colors and return a single colored point cloud in the ego frame
     of the LiDAR timestamp (interpretable for visualization).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion


CAMERAS_360: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass
class ColorizationConfig:
    min_depth: float = 1.0
    border_margin_px: int = 4
    max_incidence_cos: float = 0.15
    image_pixel_bucket: int = 1
    n_sweeps: int = 1
    output_frame: str = "ego_lidar"
    filter_dynamic_points: bool = False
    dynamic_category_prefixes: tuple[str, ...] = (
        "vehicle.",
        "human.",
        "animal.",
    )


@dataclass
class CameraProjection:
    channel: str
    image: np.ndarray
    intrinsic: np.ndarray
    pose_translation: np.ndarray
    pose_rotation: Quaternion
    calib_translation: np.ndarray
    calib_rotation: Quaternion


def _load_camera(nusc: NuScenes, sample_data_token: str) -> CameraProjection:
    sd = nusc.get("sample_data", sample_data_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    image_path = Path(nusc.dataroot) / sd["filename"]
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load camera image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return CameraProjection(
        channel=sd["channel"],
        image=rgb,
        intrinsic=np.asarray(cs["camera_intrinsic"], dtype=np.float64),
        pose_translation=np.asarray(pose["translation"], dtype=np.float64),
        pose_rotation=Quaternion(pose["rotation"]),
        calib_translation=np.asarray(cs["translation"], dtype=np.float64),
        calib_rotation=Quaternion(cs["rotation"]),
    )


def _lidar_world_pose(
    nusc: NuScenes, lidar_sd: dict
) -> tuple[np.ndarray, Quaternion, np.ndarray, Quaternion]:
    cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    return (
        np.asarray(cs["translation"], dtype=np.float64),
        Quaternion(cs["rotation"]),
        np.asarray(pose["translation"], dtype=np.float64),
        Quaternion(pose["rotation"]),
    )


def _transform_to_camera(
    points_xyz: np.ndarray,
    lidar_calib_t: np.ndarray,
    lidar_calib_r: Quaternion,
    lidar_pose_t: np.ndarray,
    lidar_pose_r: Quaternion,
    cam: CameraProjection,
) -> np.ndarray:
    """Apply LIDAR -> ego(t_lidar) -> global -> ego(t_cam) -> camera."""
    pts = points_xyz.copy()
    pts = lidar_calib_r.rotation_matrix @ pts
    pts = pts + lidar_calib_t[:, None]
    pts = lidar_pose_r.rotation_matrix @ pts
    pts = pts + lidar_pose_t[:, None]
    pts = pts - cam.pose_translation[:, None]
    pts = cam.pose_rotation.rotation_matrix.T @ pts
    pts = pts - cam.calib_translation[:, None]
    pts = cam.calib_rotation.rotation_matrix.T @ pts
    return pts


def _project(points_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = points_cam[2, :]
    uv = intrinsic @ points_cam
    z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    return uv[:2, :] / z_safe


def _quality_score(
    uv: np.ndarray, depth: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    h, w = image_shape
    cx, cy = w * 0.5, h * 0.5
    dx = (uv[0, :] - cx) / (w * 0.5)
    dy = (uv[1, :] - cy) / (h * 0.5)
    center_score = 1.0 - np.clip(np.sqrt(dx * dx + dy * dy), 0.0, 1.0)
    border_x = np.minimum(uv[0, :], w - 1 - uv[0, :]) / (w * 0.5)
    border_y = np.minimum(uv[1, :], h - 1 - uv[1, :]) / (h * 0.5)
    border_score = np.clip(np.minimum(border_x, border_y), 0.0, 1.0)
    depth_score = 1.0 / (1.0 + depth)
    return 0.5 * center_score + 0.3 * border_score + 0.2 * depth_score


def _occlusion_mask(
    uv: np.ndarray, depth: np.ndarray, image_shape: tuple[int, int], bucket: int
) -> np.ndarray:
    """Image-space z-buffer: keep only nearest depth per (bucketed) pixel."""
    h, w = image_shape
    bucket = max(1, int(bucket))
    cols = (w + bucket - 1) // bucket
    u = np.clip(np.round(uv[0, :] / bucket).astype(np.int64), 0, cols - 1)
    v = np.clip(np.round(uv[1, :] / bucket).astype(np.int64), 0, (h + bucket - 1) // bucket - 1)
    keys = v * cols + u
    order = np.argsort(depth, kind="stable")
    keys_sorted = keys[order]
    _, first_idx = np.unique(keys_sorted, return_index=True)
    keep_sorted = np.zeros(depth.shape[0], dtype=bool)
    keep_sorted[first_idx] = True
    keep = np.zeros(depth.shape[0], dtype=bool)
    keep[order] = keep_sorted
    return keep


def _is_dynamic_category(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _dynamic_keep_mask(
    nusc: NuScenes,
    lidar_sd_token: str,
    points_lidar: np.ndarray,
    dynamic_category_prefixes: tuple[str, ...],
) -> np.ndarray:
    """Keep only points not inside dynamic boxes in the lidar sensor frame."""
    if points_lidar.shape[1] == 0:
        return np.zeros(0, dtype=bool)

    _, boxes, _ = nusc.get_sample_data(lidar_sd_token)
    dynamic_boxes = [
        box
        for box in boxes
        if _is_dynamic_category(
            getattr(box, "name", "") or "", tuple(dynamic_category_prefixes)
        )
    ]
    if not dynamic_boxes:
        return np.ones(points_lidar.shape[1], dtype=bool)

    dynamic_mask = np.zeros(points_lidar.shape[1], dtype=bool)
    for box in dynamic_boxes:
        dynamic_mask |= points_in_box(box, points_lidar)
    return ~dynamic_mask


def colorize_sample(
    nusc: NuScenes,
    sample_token: str,
    cameras: Iterable[str] = CAMERAS_360,
    cfg: ColorizationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz, rgb01, source_camera_index) for colorized points."""
    cfg = cfg or ColorizationConfig()
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

    pc, _ = LidarPointCloud.from_file_multisweep(
        nusc,
        sample,
        chan="LIDAR_TOP",
        ref_chan="LIDAR_TOP",
        nsweeps=max(1, int(cfg.n_sweeps)),
        min_distance=cfg.min_depth,
    )
    points_lidar = pc.points[:3, :].astype(np.float64)

    if cfg.filter_dynamic_points:
        keep_mask = _dynamic_keep_mask(
            nusc=nusc,
            lidar_sd_token=lidar_sd["token"],
            points_lidar=points_lidar,
            dynamic_category_prefixes=cfg.dynamic_category_prefixes,
        )
        points_lidar = points_lidar[:, keep_mask]

    n_points = points_lidar.shape[1]

    lidar_calib_t, lidar_calib_r, lidar_pose_t, lidar_pose_r = _lidar_world_pose(
        nusc, lidar_sd
    )

    points_ego_lidar = (
        lidar_calib_r.rotation_matrix @ points_lidar + lidar_calib_t[:, None]
    )

    best_score = np.full(n_points, -np.inf, dtype=np.float64)
    best_color = np.zeros((n_points, 3), dtype=np.float32)
    best_cam_idx = np.full(n_points, -1, dtype=np.int32)

    cam_list = list(cameras)
    for cam_idx, cam_channel in enumerate(cam_list):
        cam_sd_token = sample["data"][cam_channel]
        cam = _load_camera(nusc, cam_sd_token)
        h, w = cam.image.shape[:2]

        pts_cam = _transform_to_camera(
            points_lidar,
            lidar_calib_t,
            lidar_calib_r,
            lidar_pose_t,
            lidar_pose_r,
            cam,
        )
        depth = pts_cam[2, :]
        uv = _project(pts_cam, cam.intrinsic)

        in_front = depth > cfg.min_depth
        in_bounds = (
            (uv[0, :] >= cfg.border_margin_px)
            & (uv[0, :] < w - cfg.border_margin_px)
            & (uv[1, :] >= cfg.border_margin_px)
            & (uv[1, :] < h - cfg.border_margin_px)
        )
        norm = np.linalg.norm(pts_cam, axis=0)
        cos_angle = np.where(norm > 1e-6, depth / norm, 0.0)
        good_angle = cos_angle >= cfg.max_incidence_cos

        valid = in_front & in_bounds & good_angle
        if not np.any(valid):
            continue

        valid_idx = np.flatnonzero(valid)
        uv_v = uv[:, valid_idx]
        depth_v = depth[valid_idx]

        keep = _occlusion_mask(uv_v, depth_v, (h, w), cfg.image_pixel_bucket)
        kept_idx = valid_idx[keep]
        if kept_idx.size == 0:
            continue
        uv_k = uv[:, kept_idx]
        depth_k = depth[kept_idx]

        u_int = np.clip(np.round(uv_k[0, :]).astype(np.int32), 0, w - 1)
        v_int = np.clip(np.round(uv_k[1, :]).astype(np.int32), 0, h - 1)
        colors = cam.image[v_int, u_int].astype(np.float32) / 255.0

        scores = _quality_score(uv_k, depth_k, (h, w))

        better = scores > best_score[kept_idx]
        upd = kept_idx[better]
        best_score[upd] = scores[better]
        best_color[upd] = colors[better]
        best_cam_idx[upd] = cam_idx

    colored_mask = best_cam_idx >= 0
    if cfg.output_frame == "ego_lidar":
        out_xyz = points_ego_lidar[:, colored_mask].T
    elif cfg.output_frame == "lidar":
        out_xyz = points_lidar[:, colored_mask].T
    else:
        raise ValueError(f"Unknown output_frame: {cfg.output_frame}")

    return out_xyz, best_color[colored_mask], best_cam_idx[colored_mask]
