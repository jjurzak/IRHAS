import json
import math
import os

import cv2
import DracoPy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CAMERA = "camera_front_wide_120fov"

df_intr = pd.read_parquet(
    f"{ROOT}/data/raw/calibration/calibration/camera_intrinsics.offline/"
    "camera_intrinsics.offline.chunk_0000.parquet"
)
df_extr = pd.read_parquet(
    f"{ROOT}/data/raw/calibration/calibration/sensor_extrinsics.offline/"
    "sensor_extrinsics.offline.chunk_0000.parquet"
)

output_dir = f"{ROOT}/data/processed/multi_clip_videos"
os.makedirs(output_dir, exist_ok=True)

video_dir = f"{ROOT}/data/raw/camera_front_wide"
obs_dir = f"{ROOT}/data/raw/obstacles"
lidar_dir = f"{ROOT}/data/raw/lidar/lidar_top_360fov.chunk_0000"

videos = set(f.split('.')[0] for f in os.listdir(video_dir) if f.endswith('.mp4'))
obstacles = set(f.split('.')[0] for f in os.listdir(obs_dir) if f.endswith('.parquet'))
lidars = set(f.split('.')[0] for f in os.listdir(lidar_dir) if f.endswith('.parquet'))
complete = sorted(videos & obstacles & lidars)[:10]

print(f"Processing {len(complete)} clips...")

CLASS_COLORS = {
    "person": (0, 0, 255),
    "rider": (0, 0, 255),
    "automobile": (0, 200, 200),
    "heavy_truck": (0, 180, 180),
    "bus": (0, 160, 160),
    "train_or_tram_car": (100, 100, 0),
    "protruding_object": (0, 100, 255),
}


def get_camera_calib(uuid):
    intr_row = df_intr.loc[(uuid, CAMERA)]
    params = json.loads(intr_row['model_parameters'])
    W, H = params['resolution']
    pp = np.array(params['principal_point'])
    poly = np.array(params['angle_to_pixeldist_poly'])
    max_angle = params.get('max_angle', math.pi)

    extr_row = df_extr.loc[(uuid, CAMERA)]
    cam_quat = [extr_row['qx'], extr_row['qy'], extr_row['qz'], extr_row['qw']]
    cam_pos = np.array([extr_row['x'], extr_row['y'], extr_row['z']])
    R_cam_from_rig = Rotation.from_quat(cam_quat).inv()

    return W, H, pp, poly, max_angle, cam_pos, R_cam_from_rig


def project_point(point_rig, pp, poly, max_angle, cam_pos, R_cam_from_rig):
    p_cam = R_cam_from_rig.apply(point_rig - cam_pos)
    x, y, z = p_cam
    if z <= 0.1:
        return None
    r = math.sqrt(x*x + y*y)
    theta = math.atan2(r, z)
    if theta > max_angle:
        return None
    pixel_dist = sum(c * (theta ** i) for i, c in enumerate(poly))
    if r < 1e-9:
        return (int(pp[0]), int(pp[1]))
    px = pp[0] + pixel_dist * (x / r)
    py = pp[1] + pixel_dist * (y / r)
    return (int(px), int(py))


def decode_lidar_spin(draco_bytes):
    try:
        pc = DracoPy.decode(draco_bytes)
        pts = np.array(pc.points, dtype=np.float32).reshape(-1, 3)
        return pts
    except Exception:
        return None


def get_lidar_extrinsics(uuid):
    try:
        row = df_extr.loc[(uuid, "lidar_top_360fov")]
        cam_quat = [row['qx'], row['qy'], row['qz'], row['qw']]
        cam_pos = np.array([row['x'], row['y'], row['z']])
        R = Rotation.from_quat(cam_quat)
        return cam_pos, R
    except KeyError:
        return None, None


for uuid in complete:
    print(f"\n[{uuid[:8]}] loading data...")

    try:
        W, H, pp, poly, max_angle, cam_pos, R_cam_from_rig = get_camera_calib(uuid)
    except KeyError:
        print(f"  no calibration, skip")
        continue

    lidar_pos, R_lidar = get_lidar_extrinsics(uuid)

    df_obs = pd.read_parquet(f"{obs_dir}/{uuid}.obstacle.offline.parquet")
    df_lidar = pd.read_parquet(f"{lidar_dir}/{uuid}.lidar_top_360fov.parquet")
    obs_ts = df_obs['timestamp_us'].values.astype(np.int64)

    video_path = f"{video_dir}/{uuid}.{CAMERA}.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = f"{output_dir}/{uuid[:8]}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    WINDOW_US = 50_000
    lidar_spins = df_lidar.sort_values('spin_start_timestamp').reset_index(drop=True)

    print(f"  {total_frames} frames, writing video...")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_t_us = int((frame_idx / fps) * 1_000_000)

        # obstacles
        mask = (obs_ts >= frame_t_us - WINDOW_US) & (obs_ts <= frame_t_us + WINDOW_US)
        snapshot = df_obs[mask].sort_values('timestamp_us').groupby('track_id').tail(1)

        for _, obj in snapshot.iterrows():
            center = np.array([obj.center_x, obj.center_y, obj.center_z])
            dist = math.sqrt(obj.center_x**2 + obj.center_y**2)
            pixel = project_point(center, pp, poly, max_angle, cam_pos, R_cam_from_rig)
            if pixel is None:
                continue
            px, py = pixel
            if px < 0 or px >= W or py < 0 or py >= H:
                continue
            color = CLASS_COLORS.get(obj.label_class, (200, 200, 0))
            size = max(6, int(200 / max(dist, 1)))
            cv2.circle(frame, (px, py), size, color, 2)
            cv2.putText(frame, f"{obj.label_class[:4]} {dist:.0f}m",
                        (px + size + 2, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # lidar
        if lidar_pos is not None:
            spin_idx = lidar_spins[lidar_spins['spin_start_timestamp'] <= frame_t_us].index
            if len(spin_idx) > 0:
                spin_row = lidar_spins.loc[spin_idx[-1]]
                pts = decode_lidar_spin(spin_row['draco_encoded_pointcloud'])
                if pts is not None:
                    pts_rig = R_lidar.apply(pts) + lidar_pos
                    step = max(1, len(pts_rig) // 2000)
                    for pt in pts_rig[::step]:
                        dist_pt = math.sqrt(pt[0]**2 + pt[1]**2)
                        if dist_pt > 60 or dist_pt < 1:
                            continue
                        pixel = project_point(pt, pp, poly, max_angle, cam_pos, R_cam_from_rig)
                        if pixel is None:
                            continue
                        px, py = pixel
                        if 0 <= px < W and 0 <= py < H:
                            intensity = max(0, min(255, int(255 * (1 - dist_pt / 60))))
                            cv2.circle(frame, (px, py), 1, (0, intensity, 0), -1)

        cv2.putText(frame, f"{uuid[:8]} | t={frame_t_us/1e6:.1f}s | objs={len(snapshot)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"  saved: {out_path}")

print("\ndone")
