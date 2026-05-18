import json
import math
import os

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UUID = "0a948f59-0a06-41a2-8e20-ac3a39ff4d61"
CAMERA = "camera_front_wide_120fov"

video_path = f"{ROOT}/data/raw/camera_front_wide/{UUID}.{CAMERA}.mp4"
df_obs = pd.read_parquet(f"{ROOT}/data/raw/obstacles/{UUID}.obstacle.offline.parquet")

df_intr = pd.read_parquet(
    f"{ROOT}/data/raw/calibration/calibration/camera_intrinsics.offline/"
    "camera_intrinsics.offline.chunk_0000.parquet"
)
df_extr = pd.read_parquet(
    f"{ROOT}/data/raw/calibration/calibration/sensor_extrinsics.offline/"
    "sensor_extrinsics.offline.chunk_0000.parquet"
)

output_dir = f"{ROOT}/data/processed/projected_frames"
os.makedirs(output_dir, exist_ok=True)

intr_row = df_intr.loc[(UUID, CAMERA)]
params = json.loads(intr_row['model_parameters'])
W, H = params['resolution']
principal_point = np.array(params['principal_point'])
angle_to_pixeldist_poly = np.array(params['angle_to_pixeldist_poly'])
max_angle = params.get('max_angle', math.pi)

extr_row = df_extr.loc[(UUID, CAMERA)]
cam_quat = [extr_row['qx'], extr_row['qy'], extr_row['qz'], extr_row['qw']]
cam_pos_in_rig = np.array([extr_row['x'], extr_row['y'], extr_row['z']])
R_rig_from_cam = Rotation.from_quat(cam_quat)
R_cam_from_rig = R_rig_from_cam.inv()

print(f"Camera pos in rig: {cam_pos_in_rig}")
print(f"Camera quat: {cam_quat}")
print(f"Principal point: {principal_point}")
print(f"Resolution: {W}x{H}")
print(f"Max angle (rad): {max_angle:.3f} = {math.degrees(max_angle):.1f} deg")


def project_point(point_rig):
    p_cam = R_cam_from_rig.apply(point_rig - cam_pos_in_rig)
    x, y, z = p_cam

    if z <= 0.1:
        return None

    r = math.sqrt(x*x + y*y)
    theta = math.atan2(r, z)

    if theta > max_angle:
        return None

    pixel_dist = 0.0
    for i, c in enumerate(angle_to_pixeldist_poly):
        pixel_dist += c * (theta ** i)

    if r < 1e-9:
        return tuple(principal_point.astype(int))

    px = principal_point[0] + pixel_dist * (x / r)
    py = principal_point[1] + pixel_dist * (y / r)

    return (int(px), int(py))


def get_color(label, dist):
    if label in ('person', 'rider'):
        if dist < 15:
            return (0, 0, 255)
        elif dist < 30:
            return (0, 165, 255)
        return (0, 255, 255)
    if dist < 10:
        return (0, 0, 255)
    if dist < 30:
        return (0, 200, 200)
    return (0, 255, 0)


print("\nopening video...")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video: {total_frames} frames @ {fps} fps")

obs_ts = df_obs['timestamp_us'].values.astype(np.int64)

sample_frames = list(range(0, min(total_frames, 600), 30))
print(f"projecting onto {len(sample_frames)} frames...\n")

WINDOW_US = 50_000

for frame_idx in sample_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    frame_t_us = int((frame_idx / fps) * 1_000_000)
    mask = (obs_ts >= frame_t_us - WINDOW_US) & (obs_ts <= frame_t_us + WINDOW_US)
    snapshot = df_obs[mask]

    snapshot = snapshot.sort_values('timestamp_us').groupby('track_id').tail(1)

    drawn = 0
    for _, obj in snapshot.iterrows():
        center = np.array([obj.center_x, obj.center_y, obj.center_z])
        dist = math.sqrt(obj.center_x**2 + obj.center_y**2)

        pixel = project_point(center)
        if pixel is None:
            continue

        px, py = pixel
        if px < 0 or px >= W or py < 0 or py >= H:
            continue

        color = get_color(obj.label_class, dist)
        size = max(6, int(200 / max(dist, 1)))
        cv2.circle(frame, (px, py), size, color, 2)
        cv2.putText(frame, f"{obj.label_class[:6]} {dist:.0f}m",
                    (px + size + 3, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        drawn += 1

    cv2.putText(frame, f"Frame {frame_idx} | t={frame_t_us/1e6:.1f}s | objs={len(snapshot)} | drawn={drawn}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = f"{output_dir}/frame_{frame_idx:04d}.jpg"
    cv2.imwrite(out_path, frame)
    print(f"  frame {frame_idx:>4} (t={frame_t_us/1e6:>5.1f}s): {len(snapshot)} objs in window, {drawn} drawn")

cap.release()
print(f"\ndone - saved frames to {output_dir}")
