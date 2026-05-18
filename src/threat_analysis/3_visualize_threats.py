import math
import os

import cv2
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UUID = "0a948f59-0a06-41a2-8e20-ac3a39ff4d61"

video_path = f"{ROOT}/data/raw/camera_front_wide/{UUID}.camera_front_wide_120fov.mp4"
df_obs = pd.read_parquet(f"{ROOT}/data/raw/obstacles/{UUID}.obstacle.offline.parquet")
df_ego = pd.read_parquet(f"{ROOT}/data/raw/extracted_labels/{UUID}.egomotion.offline.parquet")

output_dir = f"{ROOT}/data/processed/threat_frames"
os.makedirs(output_dir, exist_ok=True)

CLASS_WEIGHTS = {
    "person": 3.0,
    "rider": 3.0,
    "automobile": 1.5,
    "heavy_truck": 1.5,
    "bus": 1.5,
    "train_or_tram_car": 1.0,
    "protruding_object": 1.0,
}

dt_ego = df_ego['timestamp'].diff() / 1_000_000
dx = df_ego['x'].diff()
dy = df_ego['y'].diff()
speeds = (dx**2 + dy**2).pow(0.5) / dt_ego
ego_speed = speeds.median()
speed_mult = 1.0 + (ego_speed / 30.0)


def get_zone(cx, cy):
    if cx > 0 and abs(cy) < 2.0:
        return "CRITICAL", 3.0
    elif cx > 0 and abs(cy) < 5.0:
        return "WARNING", 2.0
    elif cx > 0:
        return "PERIPHERAL", 1.0
    else:
        return "BEHIND", 0.3


def compute_threat_at_timestamp(ts):
    snapshot = df_obs[df_obs['timestamp_us'] == ts]
    threats = []

    for _, row in snapshot.iterrows():
        cx, cy = row['center_x'], row['center_y']
        dist = math.sqrt(cx**2 + cy**2)
        label = row['label_class']
        track_id = row['track_id']

        track_hist = df_obs[
            (df_obs['track_id'] == track_id) & (df_obs['timestamp_us'] <= ts)
        ].sort_values('timestamp_us').tail(10)

        if len(track_hist) < 2:
            continue

        first_h = track_hist.iloc[0]
        last_h = track_hist.iloc[-1]
        d_first = math.sqrt(first_h.center_x**2 + first_h.center_y**2)
        d_last = math.sqrt(last_h.center_x**2 + last_h.center_y**2)
        dt_h = (last_h.timestamp_us - first_h.timestamp_us) / 1_000_000

        if dt_h == 0:
            continue

        closing = (d_first - d_last) / dt_h

        if closing > 0.1:
            ttc = d_last / closing
        else:
            ttc = float('inf')

        zone, zone_mult = get_zone(cx, cy)
        class_mult = CLASS_WEIGHTS.get(label, 1.0)

        if 0 < ttc < 100:
            ttc_factor = 1.0 / ttc
        else:
            ttc_factor = 0.0

        score = zone_mult * class_mult * speed_mult * ttc_factor

        if score > 0.05:
            threats.append({
                "track_id": track_id,
                "label": label,
                "dist": round(dist, 1),
                "ttc": round(ttc, 1),
                "zone": zone,
                "score": round(score, 2),
                "closing_kmh": round(closing * 3.6, 1),
            })

    threats.sort(key=lambda x: x['score'], reverse=True)
    return threats


def draw_threats_on_frame(frame, threats, frame_idx, ts):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (500, 40 + len(threats[:5]) * 35), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"Frame {frame_idx} | Threats: {len(threats)}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    colors = {
        "CRITICAL": (0, 0, 255),
        "WARNING": (0, 165, 255),
        "PERIPHERAL": (0, 255, 255),
        "BEHIND": (128, 128, 128),
    }

    for i, t in enumerate(threats[:5]):
        y = 65 + i * 35
        color = colors.get(t['zone'], (255, 255, 255))
        text = (
            f"{t['label'][:8]} | {t['dist']}m | "
            f"TTC:{t['ttc']}s | {t['zone']} | score:{t['score']}"
        )
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if threats and threats[0]['score'] > 1.0:
        border_color = (0, 0, 255)
    elif threats and threats[0]['score'] > 0.3:
        border_color = (0, 165, 255)
    else:
        border_color = (0, 255, 0)

    cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, 4)

    return frame


print("opening video...")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

obs_timestamps = sorted(df_obs['timestamp_us'].unique())

sample_frames = list(range(0, min(total_frames, 600), 15))

print(f"video: {total_frames} frames @ {fps} fps")
print(f"sampling {len(sample_frames)} frames...")

saved = 0
for frame_idx in sample_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    frame_time_us = int((frame_idx / fps) * 1_000_000)
    closest_ts = min(obs_timestamps, key=lambda t: abs(int(t) - frame_time_us))

    threats = compute_threat_at_timestamp(closest_ts)
    frame_out = draw_threats_on_frame(frame, threats, frame_idx, closest_ts)

    out_path = f"{output_dir}/frame_{frame_idx:04d}_threats.jpg"
    cv2.imwrite(out_path, frame_out)
    saved += 1

    if threats:
        print(f"  frame {frame_idx:>4} | top threat: {threats[0]['label']} "
              f"score={threats[0]['score']} TTC={threats[0]['ttc']}s")

cap.release()
print(f"\ndone — saved {saved} frames to {output_dir}")
