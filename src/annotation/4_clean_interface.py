import cv2
from ultralytics import YOLO


model_path = r"F:\Programy\master-thesis\IRHAS\runs\detect\train-5\weights\best.pt"
test_image_path = r"F:\Programy\master-thesis\IRHAS\data\yolo_dataset\images\raw_frames\frame_0025.jpg" 

model = YOLO(model_path)
frame = cv2.imread(test_image_path)
height, width, _ = frame.shape

cutoff_y = int(height * 0.65)
frame[cutoff_y:, :] = (0, 0, 0)


results = model.predict(
    source=frame,
    conf=0.45,         
    iou=0.3,           
    agnostic_nms=True, 
    save=False
)

annotated_frame = results[0].plot()
output_test_path = "TEST_WYNIK_CZYSTY_V2.jpg"
cv2.imwrite(output_test_path, annotated_frame)

print(f"\n{output_test_path}")