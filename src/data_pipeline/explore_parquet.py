from pandas.io import parquet
import pandas as pd 

parquet_path = "data/raw/extracted_labels/0a948f59-0a06-41a2-8e20-ac3a39ff4d61.egomotion.offline.parquet"

try:
    print(f'wczytaj plik: {parquet_path}')
    df = pd.read_parquet(parquet_path)

    print("\ncols")
    for col in df.columns:
        print(col)
    print("\nhead")
    
    print(df.head())
    
except Exception as e:
    print(f"Error: {e}")

import pandas as pd

boxes_path = "data/raw/camera_front_wide/0a948f59-0a06-41a2-8e20-ac3a39ff4d61.camera_front_wide_120fov.blurred_boxes.parquet"

df = pd.read_parquet(boxes_path)
print(df.columns)
print(df.head(3))