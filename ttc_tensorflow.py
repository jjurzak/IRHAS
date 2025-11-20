from threading import current_thread
import cv2 
import numpy as np 
import tensorflow as tf 
import tensorflow_hub as hub 
from ultralytics import YOLO 
import datetime 
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow wykrył GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("❌ TensorFlow NIE widzi GPU! Sprawdź sterowniki CUDA/cuDNN.")
    exit() 


if torch.cuda.is_available():
    print(f"✅ PyTorch (YOLO) wykrył GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ PyTorch NIE widzi GPU!")
    exit()






VIDEO_PATH = "sample_vid.mp4"
LOG_DIR = "logs/ttc_analysis" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")

file_writer = tf.summary.create_file_writer(LOG_DIR)

print("Loading MiDaS model from TF HUB...")

midas_url = "https://tfhub.dev/intel/midas/v2_1_small/1"
midas_model = hub.load(midas_url, tags=['serve'])
midas_fn = midas_model.signatures['serving_default']

yolo = YOLO("yolov8n.pt")

def run_depth_tf(frame):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(img, [256, 256], method='bicubic')
    img = img / 255.0
    img = tf.expand_dims(img, 0)
    
    img = tf.transpose(img, [0, 3, 1, 2])

    outputs = midas_fn(img)
    output = outputs['default']

    output = tf.expand_dims(output, -1)

    prediction = tf.image.resize(output, frame.shape[:2])
    depth_map = prediction.numpy().squeeze()

    return depth_map

def calculate_ttc(current_dist, prev_dist, fps=30):
    if prev_dist is None: return 999.0

    relative_speed = current_dist - prev_dist

    if relative_speed <= 0.05:
        return 999.0

    ttc_index = current_dist / relative_speed
    return ttc_index

class Smoother:
    def __init__(self, alpha=0.1):
        self.value = None
        self.alpha = alpha

    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.value * (1 - self.alpha) + new_val * self.alpha
        return self.value

cap = cv2.VideoCapture(VIDEO_PATH)
step = 0 
last_depth_val = None

depth_smoother = Smoother(alpha=0.1) 
ttc_smoother = Smoother(alpha=0.05)  


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    
    frame_view = cv2.resize(frame, (640, 360))

    h_curr, w_curr = frame_view.shape[:2]


    hood_height = 70 

    cv2.rectangle(frame_view, (0, h_curr - hood_height), (w_curr, h_curr), (0, 0, 0), -1)

    
    depth_map = run_depth_tf(frame_view)

    
    results = yolo(frame_view, verbose=False, device=0)

    height, width = frame_view.shape[:2]
    center_x = width // 2 
    
    
    raw_threat_level = 0.0
    target_found = False
    
    

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in [0, 1, 2, 3, 5,7]: 
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                obj_center = (x1+x2) // 2 
                
                    if abs(obj_center - center_x ) < width * 0.2:
                        target_found = True
                    
                    
                    y1_safe, y2_safe = max(0, y1), min(height, y2)
                    x1_safe, x2_safe = max(0, x1), min(width, x2)
                    
                    roi_depth = depth_map[y1_safe:y2_safe, x1_safe:x2_safe]
                    
                    if roi_depth.size > 0:
                        
                        
                        
                        raw_depth = np.mean(roi_depth)
                        
                        
                        avg_depth_val = depth_smoother.update(raw_depth)

                        
                        ttc = calculate_ttc(avg_depth_val, last_depth_val)
                        last_depth_val = avg_depth_val

                        
                        if ttc < 100:
                            raw_threat_level = 100 / (ttc + 0.1)
                        
                        

                        
                        final_threat = ttc_smoother.update(raw_threat_level)
                        
                        color = (0, 255, 0) # Zielony
                        if final_threat > 2.0: color = (0, 165, 255) # Pomarańczowy
                        if final_threat > 5.0: color = (0, 0, 255)   # Czerwony
                        
                        cv2.rectangle(frame_view, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_view, f"Threat: {final_threat:.1f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if not target_found:
        last_depth_val = None
        raw_threat_level = 0.0
        
        ttc_smoother.update(0.0)

    
    final_threat_logged = ttc_smoother.value if ttc_smoother.value else 0

    with file_writer.as_default():
        
        tf.summary.scalar('Traffic/Threat_Level', final_threat_logged, step=step)
        tf.summary.scalar('Traffic/Raw_Depth_Value', last_depth_val if last_depth_val else 0, step=step)

        if step % 10 == 0:
            img_tensor = np.expand_dims(cv2.cvtColor(frame_view, cv2.COLOR_BGR2RGB), 0)
            
            depth_viz = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)
            depth_tensor = np.expand_dims(cv2.cvtColor(depth_viz, cv2.COLOR_BGR2RGB), 0)
            
            tf.summary.image('Video/Input', img_tensor, step=step)
            tf.summary.image('Video/Depth_Analysis', depth_tensor, step=step)

    cv2.imshow('AI driver analysis', frame_view)
    step += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
print(f'Analiza zakoczona, Uruchom: tensorboard --logdir {LOG_DIR}')
