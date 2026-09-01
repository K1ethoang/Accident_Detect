# Accident_Detect

> Training and evaluation of **YOLO11** models for **traffic accident detection in Vietnam**.

This is the **model research** part of the graduation project *"Research on Object Detection and development of a traffic surveillance system integrating a YOLO model to detect traffic accidents in Vietnam"* (University of Transport and Communications – Ho Chi Minh City Campus, 2025).

- 📄 Report: [My-Achievements / Reports / 4th-year / DATN.pdf](https://github.com/K1ethoang/My-Achievements/blob/main/Reports/4th-year/DATN.pdf)
- 🎥 Surveillance system that integrates the model: [Surveillance-Camera-System](https://github.com/K1ethoang/Surveillance-Camera-System)

---

## Goals

- Collect and build a traffic-accident dataset that reflects real Vietnamese street conditions (high motorbike density, surveillance camera angles).
- Train the 5 YOLO11 variants (`n`, `s`, `m`, `l`, `x`) and compare accuracy vs. speed to pick the variant best suited for real-time processing.
- Produce an evaluation set of metrics: `mAP@50`, `Average IoU`, `FPS`, `Latency`.

## Dataset

| Property | Value |
|---|---|
| Source | Public videos/images (Google, YouTube, Facebook), manually labeled |
| Total images | 2919 labeled images |
| Split | Train 2335 · Validation 292 · Test 292 (80/10/10) |
| Image size | 640 × 640 |
| Classes (`nc`) | 7 |
| Labels (`names`) | `accident`, `bicycle`, `bus`, `car`, `motorcycle`, `person`, `truck` |

📥 Download the dataset (Roboflow): <https://app.roboflow.com/convertyolo2voc/convert_yolo_2_voc/1>

After downloading, extract into `dataset/` following the YOLO layout:

```
dataset/
├── data.yaml            # training config (relative paths)
├── data_evaluate.yaml   # evaluation config (adjust to absolute paths for your machine)
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}
```

## Requirements

- Python **3.12.8** (see `.python-version`)
- NVIDIA GPU + CUDA for training (recommended). Setup used in the project: RTX 3050 Ti 4GB, i7‑12700H, 16GB RAM, Windows 11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python system_info.py        # print CPU/GPU/PyTorch/CUDA info
```

Pre-trained YOLO11 weights are provided in `pre-train_of_yolo/` (`yolo11n/s/m/l/x.pt`).

## Usage

### 1. Training — `train.py`

```bash
python train.py
```

Key parameters (edit in the file): `model='./yolo11x.pt'`, `epochs=10`, `batch=4`, `workers=2`, `save_period=2`.
Outputs are written to `runs/train/<name>/` (e.g. `runs/train/v11-x/weights/best.pt`). Change `name` and the source weights to train each variant `v11-n … v11-x` in turn.

### 2. Evaluation

The scripts scan every `runs/train/*/weights/best.pt` and export CSVs:

| Script | Purpose | Output |
|---|---|---|
| `evaluate/map50_iou.py` | Compute `mAP@50` and `Average IoU` on the test set (`data_evaluate.yaml`) | `evaluate/mAP50_IoU.csv` |
| `evaluate/fps_latency.py` | Measure `FPS` and `Latency` over 1000 frames of `evaluate/video_test.mp4` | `evaluate/latency_1k_frame.csv` |
| `evaluate/export_excel.py` | Aggregate multiple runs (`time_1`, `time_2`, `time_3`) → Excel + charts | `evaluate/summary_new.xlsx`, `evaluate/charts/*.png` |
| `evaluate/draw_image.py` | Plot a 4-metric comparison of the variants | `evaluate/yolov11_comparison.png` |

```bash
python evaluate/map50_iou.py
python evaluate/fps_latency.py
python evaluate/export_excel.py
```

> `evaluate/video_test.mp4` and the images under `dataset/` are stored with Git LFS – clone with LFS if you need the real data.

### 3. Run detection on a video — `detect/yolo_detect.py`

```bash
python detect/yolo_detect.py
```

Reads `evaluate/video_test.mp4`, runs every variant found in `runs/train/`, and writes annotated videos to `detect/outputs/<version>_output.mp4`.
Drawing rules: **red** box for `accident`; **green** box for other objects only when they intersect the accident region; object counts are shown in the top-right corner.

`detect/extract_frame.py` extracts frames from the output videos (10 fps by default) for illustration.

## Results (averaged over 3 training runs, 10 epochs)

| Metric | Nano | Small | Medium | Large | X‑Large |
|---|---|---|---|---|---|
| mAP@50 – `accident` label | 0.9399 | **0.9510** | 0.9360 | 0.9304 | 0.8894 |
| mAP@50 – all labels | 0.7347 | **0.8454** | 0.7252 | 0.7275 | 0.6876 |
| Average IoU – `accident` | 0.8667 | 0.8598 | 0.8710 | **0.8731** | 0.8518 |
| Average IoU – all labels | 0.8424 | 0.8412 | 0.8449 | **0.8472** | 0.8300 |
| FPS (per 1000 frames) | **60.33** | 52.06 | 44.73 | 33.84 | 21.46 |
| Latency (s) | **0.0166** | 0.0192 | 0.0224 | 0.0295 | 0.0466 |

**Conclusion:** the **Small** variant offers the best balance of accuracy and speed; **Nano** is preferable when speed is the priority; the larger variants (`l`, `x`) do not improve accuracy enough to justify their cost.

## Directory layout

```
Accident_Detect/
├── train.py                  # YOLO11 training
├── system_info.py            # print hardware/software info
├── requirements.txt
├── dataset/                  # dataset (YOLO format) + data.yaml
├── pre-train_of_yolo/        # pre-trained weights yolo11n/s/m/l/x.pt
├── runs/train/               # per-variant training outputs (best.pt, charts, confusion matrix)
├── detect/
│   ├── yolo_detect.py        # video inference
│   ├── extract_frame.py      # frame extraction
│   └── outputs/              # result videos
└── evaluate/
    ├── map50_iou.py · fps_latency.py · export_excel.py · draw_image.py
    ├── time_1/ time_2/ time_3/   # per-run CSVs
    └── charts/                    # aggregated charts
```

## License

For educational and research purposes only.
