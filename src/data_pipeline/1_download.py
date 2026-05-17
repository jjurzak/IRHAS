import os
import zipfile

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles"

raw_dir = "./data/raw"
os.makedirs(raw_dir, exist_ok=True)

telemetry_zip = f"{raw_dir}/labels/egomotion.offline/egomotion.offline.chunk_0000.zip"

if os.path.exists(telemetry_zip):
    print("telemetry already downloaded, skip")
    telemetry_path = telemetry_zip
else:
    print("downloading telemetry/egomotion...")
    telemetry_path = hf_hub_download(
        repo_id=repo_id,
        filename="labels/egomotion.offline/egomotion.offline.chunk_0000.zip",
        repo_type="dataset",
        token=hf_token,
        local_dir=raw_dir
    )
    print("downloaded telemetry")

video_zip = f"{raw_dir}/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip"  # noqa: E501

if os.path.exists(video_zip):
    print("video already downloaded, skip")
    video_zip_path = video_zip
else:
    print("downloading video/chunk...")
    video_zip_path = hf_hub_download(
        repo_id=repo_id,
        filename="camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip",
        repo_type="dataset",
        token=hf_token,
        local_dir=raw_dir
    )
    print("downloaded video")

telemetry_extract_dir = f"{raw_dir}/extracted_labels"
os.makedirs(telemetry_extract_dir, exist_ok=True)

if os.listdir(telemetry_extract_dir):
    print("telemetry already extracted, skip")
else:
    print("extracting telemetry...")
    with zipfile.ZipFile(telemetry_path, 'r') as zip_ref:
        zip_ref.extractall(telemetry_extract_dir)
    print("telemetry extracted")

video_extract_dir = f"{raw_dir}/extracted_video"
os.makedirs(video_extract_dir, exist_ok=True)

if os.listdir(video_extract_dir):
    print("video already extracted, skip")
else:
    print("extracting video...")
    with zipfile.ZipFile(video_zip_path, 'r') as zip_ref:
        zip_ref.extractall(video_extract_dir)
    print("video extracted")

print("\ndone")
