import math
import pandas as pd

df = pd.read_parquet(
    r'..\..\data\raw\obstacles\0a948f59-0a06-41a2-8e20-ac3a39ff4d61.obstacle.offline.parquet'
)

counts = df.groupby('timestamp_us').size()
best_ts = counts.idxmax()

snapshot = df[df['timestamp_us'] == best_ts]
print(f"Timestamp: {best_ts}, obiektów: {len(snapshot)}\n")

for _, row in snapshot.iterrows():
    center_x = row['center_x']
    center_y = row['center_y']
    label = row['label_class']

    distance = math.sqrt(center_x**2 + center_y**2)
    front_back = "przed" if center_x > 0 else "za"
    left_right = "lewo" if center_y > 0 else "prawo"

    print(f"Obiekt {label} znajduje się {round(distance, 2)} metrów od środka, {front_back} i {left_right} od środka")

print('----')

for track_id in df['track_id'].unique():
    track = df[df['track_id'] == track_id].sort_values('timestamp_us')

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
        status = f"TTC: {ttc:.1f}s ⚠️"
    else:
        ttc = float('inf')
        status = "oddala się ✓"

    label = first['label_class']
    print(f"track {track_id:>3} | {label:<15} | {d_start:.1f}m → {d_end:.1f}m | closing: {closing_speed:.1f} m/s ({closing_speed*3.6:.0f} km/h) | {status}")
