import cv2
from ultralytics import YOLO

# --- KONFIGURACJA ---
model_path = r"F:\Programy\master-thesis\IRHAS\runs\detect\train-5\weights\best.pt"
test_image_path = r"F:\Programy\master-thesis\IRHAS\data\yolo_dataset\images\raw_frames\frame_0025.jpg" # Środek nagrania

model = YOLO(model_path)
frame = cv2.imread(test_image_path)
height, width, _ = frame.shape

# 1. AGRESYWNIEJSZE ROI (Odcinamy dolne 35% ekranu, gdzie jest maska pojazdu)
cutoff_y = int(height * 0.65)
frame[cutoff_y:, :] = (0, 0, 0)

# 2. DODAJEMY agnostic_nms=True
results = model.predict(
    source=frame,
    conf=0.45,         # Lekko obniżamy próg, żeby nie zgubić pieszych w tle
    iou=0.3,           
    agnostic_nms=True, # KLUCZ DO SUKCESU! Wymusza łączenie ramek niezależnie od ID
    save=False
)

annotated_frame = results[0].plot()
output_test_path = "TEST_WYNIK_CZYSTY_V2.jpg"
cv2.imwrite(output_test_path, annotated_frame)

print(f"\nGOTOWE! Otwórz plik: {output_test_path}")