import os

from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from ultralytics import YOLO

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

input_frames_dir = f"{ROOT}/data/yolo_dataset/images/raw_frames"
output_dataset_dir = f"{ROOT}/data/yolo_dataset/labeled_data"
os.makedirs(output_dataset_dir, exist_ok=True)

ontology = CaptionOntology({
    "car": "Vehicle",
    "truck": "Vehicle",
    "bus": "Vehicle",
    "motorcycle": "Vehicle",
    "person": "VRU",
    "pedestrian": "VRU",
    "bicycle": "VRU",
    "cyclist": "VRU",
    "traffic light": "Traffic_Signal",
    "traffic sign": "Traffic_Signal",
    "stop sign": "Traffic_Signal",
    "speed limit sign": "Traffic_Signal",
    "traffic cone": "Obstacle",
    "barrier": "Obstacle",
    "guardrail": "Obstacle",
    "debris": "Obstacle",
})

print("1. loading grounding dino...")
base_model = GroundingDINO(ontology=ontology)

print("\n2. auto-labeling...")
base_model.label(
    input_folder=input_frames_dir,
    extension=".jpg",
    output_folder=output_dataset_dir,
)

print("\n3. training yolo...")
dataset_yaml = f"{output_dataset_dir}/data.yaml"
model = YOLO("yolov8n.pt")
model.train(
    data=dataset_yaml,
    epochs=20,
    batch=8,
    imgsz=1280,
    device=0,
    workers=0,
)