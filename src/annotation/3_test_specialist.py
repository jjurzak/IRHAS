import cv2
from ultralytics import YOLO

model_path = r"F:\Programy\master-thesis\IRHAS\runs\detect\train-5\weights\best.pt"

test_image_path = r"F:\Programy\master-thesis\IRHAS\data\yolo_dataset\images\raw_frames\frame_0025.jpg" # Środek nagrania

print(f"Ładowanie modelu z: {model_path}...")
model = YOLO(model_path)

print("Uruchamianie inferencji (detekcji)...")
results = model.predict(
    source=test_image_path,
    conf=0.3,  
    save=False 
)

result = results[0]

annotated_frame = result.plot()

output_test_path = "TEST_WYNIK_IRHAS.jpg"
cv2.imwrite(output_test_path, annotated_frame)

print(f"\nGOTOWE! Zobacz wygenerowane zdjęcie: {output_test_path}")

print("\n--- ZNALEZIONE OBIEKTY ---")
for box in result.boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    print(f"- Znalazłem: {class_name} (Pewność: {confidence*100:.1f}%)")