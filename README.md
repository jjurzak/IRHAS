
```
██╗██████╗ ██╗  ██╗ █████╗ ███████╗
██║██╔══██╗██║  ██║██╔══██╗██╔════╝
██║██████╔╝███████║███████║███████╗
██║██╔══██╗██╔══██║██╔══██║╚════██║
██║██║  ██║██║  ██║██║  ██║███████║
╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
Intelligent Road Hazard Analysis System
```
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Status](https://img.shields.io/badge/status-WIP-yellow)
![Performance](https://img.shields.io/badge/FPS-31--35-blue)
![YOLO](https://img.shields.io/badge/YOLO-v8n-orange)
![Transformer](https://img.shields.io/badge/Transformer-enabled-purple)
![CUDA](https://img.shields.io/badge/CUDA-12.1-informational)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Hardware](https://img.shields.io/badge/GPU-RTX_4070_Ti-red)
![License](https://img.shields.io/badge/license-private-lightgrey)

IRHAS is a modular real-time road hazard analysis system that fuses object detection and vehicle motion data to assess driving threats.

---

## ✨ Key Features
- 🔍 Detection of vehicles, pedestrians, traffic lights, bicycles, and obstacles
- ⚡ *Parallel* inference of *N* YOLOv8n models*
- 🧠 Detection fusion powered by a transformer module
- 🚘 Ego-motion integration (speed, acceleration, steering, curvature)
- 🚨 Automatic alert levels: SAFE / WARNING / CRITICAL
- 🎥 Real-time overlay visualization on video frames

---

## 🧩 IRHAS Architecture (ASCII DIAGRAM)
```
                       ┌───────────────────────────┐
                       │      VIDEO INPUT (FHD)    │
                       └───────────────┬───────────┘
                                       │
                              ┌────────▼────────┐
                              │ PREPROCESSING   │
                              │ 1280×720 resize │
                              └────────┬────────┘
                                       │
        ┌─────── P A R A L L E L   M O D E L   C L U S T E R ───────┐
        │                                                           │
 ┌──────▼──────┐ ┌────────▼────────┐ ┌────────▼────────┐ ┌──────────▼──────┐
 │ YOLO: CARS  │ │ YOLO: PERSON    │ │ YOLO: TRAFFIC   │ │ YOLO: OBSTACLES │
 │ specialized │ │ specialized     │ │ LIGHTS (spec)   │ │/ BICYCLES (spec)│
 └──────▲──────┘ └────────▲────────┘ └────────▲────────┘ └────────▲────────┘
        │                 │                 │                 │
        └────────────┬────┴────┬────────────┴────┬────────────┘
                     ▼         ▼                 ▼
                ┌───────────────────────────────────────┐
                │      TRANSFORMER FUSION ENGINE        │
                └─────────────────────┬─────────────────┘
                                      │ fused detections
                             ┌────────▼────────┐
                             │ EGO-MOTION DATA │ speed/accel/steer
                             └────────┬────────┘
                                      │
                             ┌────────▼────────┐
                             │  THREAT SCORE   │ zone × class × dynamics
                             └────────┬────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │ ALERT ENGINE + OVERLAY    │
                        │ SAFE / WARNING / CRIT     │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │     REAL-TIME DISPLAY     │
                        │         31–35 FPS         │
                        └───────────────────────────┘
```

---

## ⚙️ Estimated Runtime & Performance
> ⚠️ Note: Values below are **predicted** based on model size, parallel inference assumptions and hardware targets.  
> Actual performance may **very vary** depending on GPU, drivers and runtime optimization.

| Metric | Estimated Value |
|--------|-----------------|
| Input Resolution | 1920×1080 (FHD) |
| Processing Resolution | 1280×720 (HD) |
| Latency | ~28–32 ms / frame |
| Output FPS | ~31–35 FPS |
| VRAM Usage | ~7–8 GB |
| CPU Load | ~30–40% |
| GPU Load | ~70–80% |

---

## 🎯 Why IRHAS?
- 🧩 Lightweight modular architecture: instead of one heavy multi-class model, IRHAS uses several small specialized models.
- ⚡ On-demand activation: we load and run only the model(s) required for the current scene - minimizing concurrent memory and compute usage.
- 🪶 Low resource footprint: fewer simultaneous instances => lower VRAM/CPU load and reduced latency.
- 📈 Higher effectiveness at lower cost: better mAP with less runtime overhead compared to monolithic detectors.
- 🚗 Edge-ready: optimized for consumer and embedded hardware via quantization and selective inference.

---

## 🧪 Dataset & Training
- 📚 Source: PhysicalAI (HuggingFace)  
- 🤖 Auto-label workflow: Autodistill + GroundingDINO/SAM  
- 🧱 N independent YOLOv8n models trained on specialized classes*  
- 🧮 INT8 model quantization  
- 🔀 Train/val/test split: 70 / 15 / 15

---

## 🗺️ Roadmap
- [x] Dataset collection + preprocessing
- [ ] Auto annotation + validation
- [ ] YOLOv8 specialist model training
- [ ] Parallel inference implementation
- [ ] Transformer fusion + threat scoring
- [ ] CUDA/TensorRT optimization
- [ ] Edge/Jetson deployment
- [ ] Demo video, telemetry + benchmarking

---

## 📌 Project Status
⚠️ Active development



<br>
<small><sup>*</sup> Default configuration includes 4 specialist models (e.g. cars, pedestrians, traffic lights, obstacles). IRHAS supports configuring <b><i>N</i></b> models so users can add or remove specialist detectors. Models can be deployed as a fixed set (e.g., 4) or <i><b>hot-swapped</b></i> at runtime: only currently active models are loaded/executed to minimize VRAM and CPU usage. This enables flexible trade-offs between detection coverage and resource footprint; recommended default = 4 for typical edge/desktop deployments.</small>

