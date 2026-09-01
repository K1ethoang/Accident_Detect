# Accident_Detect

> Huấn luyện và đánh giá mô hình **YOLO11** cho bài toán **phát hiện tai nạn giao thông tại Việt Nam**.

Đây là phần **nghiên cứu mô hình** của đồ án tốt nghiệp *"Nghiên cứu bài toán Object Detection và phát triển hệ thống giám sát giao thông tích hợp mô hình YOLO để phát hiện tai nạn giao thông tại Việt Nam"* (Trường ĐH Giao thông Vận tải – Phân hiệu TP.HCM, 2025).

- 📄 Báo cáo: [My-Achievements / Reports / 4th-year / DATN.pdf](https://github.com/K1ethoang/My-Achievements/blob/main/Reports/4th-year/DATN.pdf)
- 🎥 Hệ thống giám sát tích hợp mô hình: [Surveillance-Camera-System](https://github.com/K1ethoang/Surveillance-Camera-System)

---

## Mục tiêu

- Thu thập và xây dựng bộ dữ liệu tai nạn giao thông phản ánh đúng đặc thù đường phố Việt Nam (mật độ xe máy cao, góc camera giám sát).
- Huấn luyện 5 biến thể YOLO11 (`n`, `s`, `m`, `l`, `x`) và so sánh độ chính xác / tốc độ để chọn biến thể phù hợp cho xử lý thời gian thực.
- Xuất bộ chỉ số đánh giá: `mAP@50`, `Average IoU`, `FPS`, `Latency`.

## Bộ dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | Video/ảnh công khai (Google, YouTube, Facebook), gán nhãn thủ công |
| Tổng số ảnh | 2919 ảnh đã gán nhãn |
| Chia tập | Train 2335 · Validation 292 · Test 292 (80/10/10) |
| Kích thước ảnh | 640 × 640 |
| Số lớp (`nc`) | 7 |
| Nhãn (`names`) | `accident`, `bicycle`, `bus`, `car`, `motorcycle`, `person`, `truck` |

📥 Tải bộ dữ liệu (Roboflow): <https://app.roboflow.com/convertyolo2voc/convert_yolo_2_voc/1>

Sau khi tải, giải nén vào thư mục `dataset/` theo cấu trúc YOLO:

```
dataset/
├── data.yaml            # cấu hình train (đường dẫn tương đối)
├── data_evaluate.yaml   # cấu hình đánh giá (sửa lại đường dẫn tuyệt đối cho phù hợp máy)
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}
```

## Yêu cầu môi trường

- Python **3.12.8** (xem `.python-version`)
- GPU NVIDIA + CUDA để huấn luyện (khuyến nghị). Cấu hình dùng trong đồ án: RTX 3050 Ti 4GB, i7‑12700H, 16GB RAM, Windows 11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python system_info.py        # in thông tin CPU/GPU/PyTorch/CUDA
```

Trọng số YOLO11 pre‑trained có sẵn trong `pre-train_of_yolo/` (`yolo11n/s/m/l/x.pt`).

## Sử dụng

### 1. Huấn luyện — `train.py`

```bash
python train.py
```

Tham số chính (sửa trực tiếp trong file): `model='./yolo11x.pt'`, `epochs=10`, `batch=4`, `workers=2`, `save_period=2`.
Kết quả lưu tại `runs/train/<name>/` (ví dụ `runs/train/v11-x/weights/best.pt`). Đổi `name` và trọng số nguồn để huấn luyện lần lượt các biến thể `v11-n … v11-x`.

### 2. Đánh giá

Các script quét toàn bộ `runs/train/*/weights/best.pt` rồi xuất CSV:

| Script | Chức năng | Kết quả |
|---|---|---|
| `evaluate/map50_iou.py` | Tính `mAP@50` và `Average IoU` trên tập test (`data_evaluate.yaml`) | `evaluate/mAP50_IoU.csv` |
| `evaluate/fps_latency.py` | Đo `FPS` và `Latency` trên 1000 khung hình của `evaluate/video_test.mp4` | `evaluate/latency_1k_frame.csv` |
| `evaluate/export_excel.py` | Gộp kết quả nhiều lần chạy (`time_1`, `time_2`, `time_3`) → Excel + biểu đồ | `evaluate/summary_new.xlsx`, `evaluate/charts/*.png` |
| `evaluate/draw_image.py` | Vẽ biểu đồ so sánh 4 chỉ số của các biến thể | `evaluate/yolov11_comparison.png` |

```bash
python evaluate/map50_iou.py
python evaluate/fps_latency.py
python evaluate/export_excel.py
```

> `evaluate/video_test.mp4` và ảnh trong `dataset/` được lưu qua Git LFS – clone kèm LFS nếu cần dữ liệu thật.

### 3. Chạy phát hiện trên video — `detect/yolo_detect.py`

```bash
python detect/yolo_detect.py
```

Đọc `evaluate/video_test.mp4`, chạy lần lượt mọi biến thể trong `runs/train/`, xuất video có bounding box vào `detect/outputs/<version>_output.mp4`.
Quy tắc vẽ: khung **đỏ** cho `accident`; khung **xanh** cho các đối tượng khác chỉ khi chúng giao nhau với vùng tai nạn; đếm số đối tượng ở góc phải trên.

`detect/extract_frame.py` trích khung hình từ các video output (mặc định 10 fps) để phục vụ minh họa.

## Kết quả (trung bình 3 lần huấn luyện, 10 epoch)

| Chỉ số | Nano | Small | Medium | Large | X‑Large |
|---|---|---|---|---|---|
| mAP@50 – nhãn `accident` | 0.9399 | **0.9510** | 0.9360 | 0.9304 | 0.8894 |
| mAP@50 – tất cả nhãn | 0.7347 | **0.8454** | 0.7252 | 0.7275 | 0.6876 |
| Average IoU – `accident` | 0.8667 | 0.8598 | 0.8710 | **0.8731** | 0.8518 |
| Average IoU – tất cả nhãn | 0.8424 | 0.8412 | 0.8449 | **0.8472** | 0.8300 |
| FPS (1000 khung/giây) | **60.33** | 52.06 | 44.73 | 33.84 | 21.46 |
| Latency (s) | **0.0166** | 0.0192 | 0.0224 | 0.0295 | 0.0466 |

**Kết luận:** biến thể **Small** cân bằng tốt nhất giữa độ chính xác và tốc độ; **Nano** phù hợp khi ưu tiên tốc độ; các biến thể lớn (`l`, `x`) không cải thiện độ chính xác tương xứng chi phí.

## Cấu trúc thư mục

```
Accident_Detect/
├── train.py                  # huấn luyện YOLO11
├── system_info.py            # in thông tin phần cứng/phần mềm
├── requirements.txt
├── dataset/                  # bộ dữ liệu (YOLO format) + data.yaml
├── pre-train_of_yolo/        # trọng số pre-trained yolo11n/s/m/l/x.pt
├── runs/train/               # kết quả huấn luyện từng biến thể (best.pt, biểu đồ, confusion matrix)
├── detect/
│   ├── yolo_detect.py        # suy luận trên video
│   ├── extract_frame.py      # trích khung hình
│   └── outputs/              # video kết quả
└── evaluate/
    ├── map50_iou.py · fps_latency.py · export_excel.py · draw_image.py
    ├── time_1/ time_2/ time_3/   # CSV của từng lần chạy
    └── charts/                    # biểu đồ tổng hợp
```

## License

Chỉ dùng cho mục đích học tập và nghiên cứu.
