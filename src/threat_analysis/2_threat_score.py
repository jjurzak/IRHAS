import math
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UUID = "0a948f59-0a06-41a2-8e20-ac3a39ff4d61"

df_obs = pd.read_parquet(f"{ROOT}/data/raw/obstacles/{UUID}.obstacle.offline.parquet")
df_ego = pd.read_parquet(f"{ROOT}/data/raw/extracted_labels/{UUID}.egomotion.offline.parquet")

CLASS_WEIGHTS = {
    "person": 3.0,
    "rider": 3.0,
    "automobile": 1.5,
    "heavy_truck": 1.5,
    "bus": 1.5,
    "train_or_tram_car": 1.0,
    "protruding_object": 1.0,
}

ego_speed = 0.0
if 'vx' in df_ego.columns:
    ego_speed = math.sqrt(df_ego['vx'].mean()**2 + df_ego['vy'].mean()**2)
else:
    dt_ego = df_ego['timestamp'].diff() / 1_000_000
    dx = df_ego['x'].diff()
    dy = df_ego['y'].diff()
    speeds = (dx**2 + dy**2).pow(0.5) / dt_ego
    ego_speed = speeds.median()

speed_mult = 1.0 + (ego_speed / 30.0)

results = []

for track_id in df_obs['track_id'].unique():
    track = df_obs[df_obs['track_id'] == track_id].sort_values('timestamp_us')

    if len(track) < 3:
        continue

    first = track.iloc[0]
    last = track.iloc[-1]

    d_start = math.sqrt(first.center_x**2 + first.center_y**2)
    d_end = math.sqrt(last.center_x**2 + last.center_y**2)
    dt = (last.timestamp_us - first.timestamp_us) / 1_000_000

    if dt == 0:
        continue

    closing_speed = (d_start - d_end) / dt

    if closing_speed > 0:
        ttc = d_end / closing_speed
    else:
        ttc = float('inf')

    cx = last.center_x
    cy = last.center_y

    if cx > 0 and abs(cy) < 2.0:
        zone = "CRITICAL"
        zone_mult = 3.0
    elif cx > 0 and abs(cy) < 5.0:
        zone = "WARNING"
        zone_mult = 2.0
    elif cx > 0:
        zone = "PERIPHERAL"
        zone_mult = 1.0
    else:
        zone = "BEHIND"
        zone_mult = 0.3

    label = last.label_class
    class_mult = CLASS_WEIGHTS.get(label, 1.0)

    if 0 < ttc < 100:
        ttc_factor = 1.0 / ttc
    else:
        ttc_factor = 0.0

    threat_score = zone_mult * class_mult * speed_mult * ttc_factor

    results.append({
        "track_id": track_id,
        "label": label,
        "d_end": round(d_end, 1),
        "closing_kmh": round(closing_speed * 3.6, 1),
        "ttc": round(ttc, 1) if ttc < 1000 else "∞",
        "zone": zone,
        "threat_score": round(threat_score, 3),
    })

results.sort(key=lambda x: x['threat_score'], reverse=True)

print(f"Ego speed: {ego_speed:.1f} m/s ({ego_speed*3.6:.0f} km/h)")
print(f"Speed multiplier: {speed_mult:.2f}")
print(f"\n{'='*80}")
print(f"{'RANK':<5} {'TRACK':<7} {'CLASS':<18} {'DIST':<7} {'CLOSE':<10} {'TTC':<8} {'ZONE':<12} {'SCORE':<8}")
print(f"{'='*80}")

for i, r in enumerate(results[:15], 1):
    print(
        f"{i:<5} {r['track_id']:<7} {r['label']:<18} {r['d_end']:<7} "
        f"{r['closing_kmh']:<10} {r['ttc']:<8} {r['zone']:<12} {r['threat_score']:<8}"
    )
