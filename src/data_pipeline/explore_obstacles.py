import pandas as pd

obstacle_path = obstacle_path = r"F:\Programy\master-thesis\IRHAS\data\raw\obstacles\0a948f59-0a06-41a2-8e20-ac3a39ff4d61.obstacle.offline.parquet"

try:
    df = pd.read_parquet(obstacle_path)
    
    print("cols")
    for col in df.columns:
        print(f"- {col}")
        
    print("\n2 przeszkody")
    print(df.head(2))
    
except Exception as e:
    print(f"Błąd przy wczytywaniu: {e}")