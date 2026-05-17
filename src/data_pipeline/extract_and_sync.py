import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

UUID = "0a948f59-0a06-41a2-8e20-ac3a39ff4d61"
egomotion_path = f"{ROOT}/data/raw/extracted_labels/{UUID}.egomotion.offline.parquet"
video_path = f"{ROOT}/data/raw/camera_front_wide/{UUID}.camera_front_wide_120fov.mp4"

output_dir = f"{ROOT}/data/yolo_dataset/images/raw_frames"
os.makedirs(output_dir, exist_ok=True)

print("1. egomotion vectors...")
df_ego = pd.read_parquet(egomotion_path)
imu_times = df_ego['timestamp'].values / 1_000_000.0

interp_x = interp1d(imu_times, df_ego['x'].values, bounds_error=False, fill_value="extrapolate")
interp_y = interp1d(imu_times, df_ego['y'].values, bounds_error=False, fill_value="extrapolate")
interp_z = interp1d(imu_times, df_ego['z'].values, bounds_error=False, fill_value="extrapolate")

quats = df_ego[['qx', 'qy', 'qz', 'qw']].values
rotations = R.from_quat(quats)
slerp = Slerp(imu_times, rotations)

print("2. cutting video and syncing...")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

max_frames = 50
frame_idx = 0
synced_metadata = []

while cap.isOpened() and frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = frame_idx / fps
    safe_time = np.clip(current_time, imu_times[0], imu_times[-1])

    x, y, z = interp_x(current_time), interp_y(current_time), interp_z(current_time)
    quat = slerp([safe_time])[0].as_quat()

    img_name = f"frame_{frame_idx:04d}.jpg"
    cv2.imwrite(f"{output_dir}/{img_name}", frame)

    synced_metadata.append({
        "image_file": img_name,
        "time_sec": round(current_time, 4),
        "ego_translation": [round(float(x), 5), round(float(y), 5), round(float(z), 5)],
        "ego_rotation_quat": [round(quat[0], 5), round(quat[1], 5), round(quat[2], 5), round(quat[3], 5)]
    })

    frame_idx += 1

cap.release()

json_path = f"{output_dir}/synced_metadata.json"
with open(json_path, "w") as f:
    json.dump(synced_metadata, f, indent=4)

print(f"\ndone - {frame_idx} frames saved to {output_dir}")
print(f"metadata: {json_path}")
