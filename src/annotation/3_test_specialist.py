import cv2
from ultralytics import YOLO

# 1. Ścieżka do NASZEGO autorskiego, wytrenowanego modelu
# Zmień "train-5" jeśli w przyszłości wygeneruje Ci inny folder
model_path = r"F:\Programy\master-thesis\IRHAS\runs\detect\train-5\weights\best.pt"

# 2. Wybieramy jedno z naszych wyciętych zdjęć
test_image_path = r"F:\Programy\master-thesis\IRHAS\data\yolo_dataset\images\raw_frames\frame_0025.jpg" # Środek nagrania

print(f"Ładowanie modelu z: {model_path}...")
model = YOLO(model_path)

print("Uruchamianie inferencji (detekcji)...")
# Predict zwraca listę wyników (dla jednego zdjęcia będzie to wynik z indeksem 0)
results = model.predict(
    source=test_image_path,
    conf=0.3,  # Pokazuj tylko obiekty, których model jest pewien na minimum 30%
    save=False # Nie zapisujemy domyślnie, zrobimy to własnoręcznie przez OpenCV
)

# Pobieramy wynik dla pierwszego (i jedynego) obrazka
result = results[0]

# Generujemy obrazek z narysowanymi ramkami
annotated_frame = result.plot()

# Zapisujemy go w głównym folderze, żebyś mógł go łatwo otworzyć i obejrzeć
output_test_path = "TEST_WYNIK_IRHAS.jpg"
cv2.imwrite(output_test_path, annotated_frame)

print(f"\nGOTOWE! Zobacz wygenerowane zdjęcie: {output_test_path}")

# Dodatkowo wypiszmy w konsoli, co znalazł:
print("\n--- ZNALEZIONE OBIEKTY ---")
for box in result.boxes:
    # Pobranie nazwy klasy i pewności (Confidence)
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    print(f"- Znalazłem: {class_name} (Pewność: {confidence*100:.1f}%)")