import json
import math
import os
import random
import shutil

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CAMERA = "camera_front_wide_120fov"
OUT_W, OUT_H = 1280, 720
SUBSAMPLE = 3
NUM_CLIPS = 30
TRAIN_RATIO = 0.8
random.seed(42)

CLASS_MAP = {
    "automobile": 0,
    "heavy_truck": 1,
    "bus": 2,
    "train_or_tram_car": 3,
    "other_vehicle": 4,
    "trailer": 5,
    "person": 6,
    "rider": 7,
    "stroller": 8,
    "animal": 9,
    "protruding_object": 10,
}

CLASS_NAMES = [
    "automobile", "heavy_truck", "bus", "train_or_tram_car",
    "other_vehicle", "trailer", "person", "rider",
    "stroller", "animal", "protruding_object",
]

video_dir = f"{ROOT}/data/raw/camera_front_wide"
obs_dir = f"{ROOT}/data/raw/obstacles"
calib_intr = f"{ROOT}/data/raw/calibration/calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet"
calib_extr = f"{ROOT}/data/raw/calibration/calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet"

out_base = f"{ROOT}/data/yolo_dataset/nvidia_labels"
for split in ("train", "val"):
    os.makedirs(f"{out_base}/images/{split}", exist_ok=True)
    os.makedirs(f"{out_base}/labels/{split}", exist_ok=True)

df_intr = pd.read_parquet(calib_intr)
df_extr = pd.read_parquet(calib_extr)

videos = set(f.split('.')[0] for f in os.listdir(video_dir) if f.endswith('.mp4'))
obstacles = set(f.split('.')[0] for f in os.listdir(obs_dir) if f.endswith('.parquet'))
complete = sorted(videos & obstacles)
selected = random.sample(complete, min(NUM_CLIPS, len(complete)))

print(f"Selected {len(selected)} clips, subsampling every {SUBSAMPLE} frames")
print(f"Target: {OUT_W}x{OUT_H}")


def get_calib(uuid):
    intr_row = df_intr.loc[(uuid, CAMERA)]
    params = json.loads(intr_row['model_parameters'])
    W_orig, H_orig = params['resolution']
    pp = np.array(params['principal_point'])
    poly = np.array(params['angle_to_pixeldist_poly'])
    max_angle = params.get('max_angle', math.pi)
    extr_row = df_extr.loc[(uuid, CAMERA)]
    cam_quat = [extr_row['qx'], extr_row['qy'], extr_row['qz'], extr_row['qw']]
    cam_pos = np.array([extr_row['x'], extr_row['y'], extr_row['z']])
    R_cam = Rotation.from_quat(cam_quat).inv()
    return W_orig, H_orig, pp, poly, max_angle, cam_pos, R_cam


def project_point(pt_rig, pp, poly, max_angle, cam_pos, R_cam):
    p_cam = R_cam.apply(pt_rig - cam_pos)
    x, y, z = p_cam
    if z <= 0.1:
        return None
    r = math.sqrt(x*x + y*y)
    theta = math.atan2(r, z)
    if theta > max_angle:
        return None
    pixel_dist = sum(c * (theta ** i) for i, c in enumerate(poly))
    if r < 1e-9:
        return np.array([pp[0], pp[1]])
    return np.array([pp[0] + pixel_dist * (x / r), pp[1] + pixel_dist * (y / r)])


def box3d_to_yolo(cx, cy, cz, sx, sy, sz, qx, qy, qz, qw,
                  pp, poly, max_angle, cam_pos, R_cam, W_orig, H_orig):
    """Project 3D box corners to 2D, return YOLO bbox (cx,cy,w,h) normalized."""
    R_obj = Rotation.from_quat([qx, qy, qz, qw])
    half = np.array([sx, sy, sz]) / 2.0
    corners_local = np.array([
        [dx * half[0], dy * half[1], dz * half[2]]
        for dx in [-1, 1] for dy in [-1, 1] for dz in [-1, 1]
    ])
    center = np.array([cx, cy, cz])
    corners_rig = R_obj.apply(corners_local) + center

    pixels = []
    for pt in corners_rig:
        px = project_point(pt, pp, poly, max_angle, cam_pos, R_cam)
        if px is not None:
            pixels.append(px)

    if len(pixels) < 2:
        return None

    pixels = np.array(pixels)
    x_min = max(0, pixels[:, 0].min())
    x_max = min(W_orig - 1, pixels[:, 0].max())
    y_min = max(0, pixels[:, 1].min())
    y_max = min(H_orig - 1, pixels[:, 1].max())

    if x_max <= x_min or y_max <= y_min:
        return None

    # scale to output resolution
    scale_x = OUT_W / W_orig
    scale_y = OUT_H / H_orig

    x_min *= scale_x
    x_max *= scale_x
    y_min *= scale_y
    y_max *= scale_y

    # YOLO format: normalized cx, cy, w, h
    box_cx = ((x_min + x_max) / 2) / OUT_W
    box_cy = ((y_min + y_max) / 2) / OUT_H
    box_w = (x_max - x_min) / OUT_W
    box_h = (y_max - y_min) / OUT_H

    if box_w < 0.002 or box_h < 0.002:
        return None

    return box_cx, box_cy, box_w, box_h


total_frames = 0
WINDOW_US = 50_000

for clip_idx, uuid in enumerate(selected):
    print(f"\n[{clip_idx+1}/{len(selected)}] {uuid[:8]}...")

    try:
        W_orig, H_orig, pp, poly, max_angle, cam_pos, R_cam = get_calib(uuid)
    except KeyError:
        print("  no calibration, skip")
        continue

    df_obs = pd.read_parquet(f"{obs_dir}/{uuid}.obstacle.offline.parquet")
    obs_ts = df_obs['timestamp_us'].values.astype(np.int64)

    video_path = f"{video_dir}/{uuid}.{CAMERA}.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    split = "train" if random.random() < TRAIN_RATIO else "val"
    saved = 0

    for frame_idx in range(0, n_frames, SUBSAMPLE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_t_us = int((frame_idx / fps) * 1_000_000)
        mask = (obs_ts >= frame_t_us - WINDOW_US) & (obs_ts <= frame_t_us + WINDOW_US)
        snapshot = df_obs[mask].sort_values('timestamp_us').groupby('track_id').tail(1)

        if len(snapshot) == 0:
            continue

        labels = []
        for _, obj in snapshot.iterrows():
            cls_id = CLASS_MAP.get(obj.label_class)
            if cls_id is None:
                continue

            bbox = box3d_to_yolo(
                obj.center_x, obj.center_y, obj.center_z,
                obj.size_x, obj.size_y, obj.size_z,
                obj.orientation_x, obj.orientation_y,
                obj.orientation_z, obj.orientation_w,
                pp, poly, max_angle, cam_pos, R_cam, W_orig, H_orig
            )
            if bbox is None:
                continue

            labels.append(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

        if not labels:
            continue

        frame_resized = cv2.resize(frame, (OUT_W, OUT_H))
        name = f"{uuid[:8]}_{frame_idx:05d}"

        cv2.imwrite(f"{out_base}/images/{split}/{name}.jpg", frame_resized,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(f"{out_base}/labels/{split}/{name}.txt", "w") as f:
            f.write("\n".join(labels))

        saved += 1

    cap.release()
    total_frames += saved
    print(f"  {split}: {saved} frames saved")

# data.yaml
yaml_path = f"{out_base}/data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"path: {out_base}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write(f"nc: {len(CLASS_NAMES)}\n")
    f.write(f"names: {CLASS_NAMES}\n")

print(f"\ndone — {total_frames} total frames")
print(f"dataset: {out_base}")
print(f"yaml: {yaml_path}")
