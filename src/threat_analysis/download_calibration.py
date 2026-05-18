import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
output_dir = f"{ROOT}/data/raw/calibration"
os.makedirs(output_dir, exist_ok=True)

repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles"

files = [
    "calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet",
    "calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet",
]

for f in files:
    print(f"downloading {f}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=f,
        repo_type="dataset",
        local_dir=output_dir,
    )

print(f"\ndone — saved to {output_dir}")
