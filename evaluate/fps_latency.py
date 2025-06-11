import os
import time
import torch
import csv
import cv2
from ultralytics import YOLO

# Đường dẫn video test
VIDEO_PATH = "./evaluate/video_test.mp4"

CHECKPOINT_DIR = "./runs/train/"  # Thư mục chứa các weight của YOLOv11

# File kết quả CSV
OUTPUT_CSV = "./evaluate/latency_1k_frame.csv"

# Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def find_checkpoint(folder_path):
    """Tìm best.pt hoặc fallback về last.pt trong thư mục model."""
    weight_path = os.path.join(folder_path, "weights")
    best_ckpt = os.path.join(weight_path, "best.pt")
    last_ckpt = os.path.join(weight_path, "last.pt")
    if os.path.exists(best_ckpt):
        return best_ckpt
    elif os.path.exists(last_ckpt):
        return last_ckpt
    else:
        return None

def process_video_yolo(model, video_path, num_frames=1000):
    """Chạy video với model YOLO và đo FPS + latency."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None

    frame_count = 0
    total_time = 0
    latencies = []

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        _ = model(frame_rgb)
        end_time = time.time()

        inference_time = end_time - start_time
        latencies.append(inference_time)
        total_time += inference_time
        frame_count += 1

    cap.release()

    if frame_count == 0:
        print("No frames processed.")
        return None, None

    avg_fps = frame_count / total_time
    avg_latency = sum(latencies) / len(latencies)
    return avg_fps, avg_latency

def main():
    # Tạo danh sách lưu kết quả
    results = []

    # Lặp qua tất cả thư mục con trong CHECKPOINT_DIR
    for subdir in os.listdir(CHECKPOINT_DIR):
        folder_path = os.path.join(CHECKPOINT_DIR, subdir)
        if not os.path.isdir(folder_path):
            continue

        ckpt_path = find_checkpoint(folder_path)
        if not ckpt_path:
            print(f"No checkpoint found in {folder_path}")
            continue

        print(f"Loading model from {ckpt_path}...")
        try:
            model = YOLO(ckpt_path)
        except Exception as e:
            print(f"Failed to load model from {ckpt_path}: {e}")
            continue

        print(f"Processing video with {subdir}...")
        fps, latency = process_video_yolo(model, VIDEO_PATH)
        if fps is not None:
            results.append({
                'Version': subdir,
                'FPS': fps,
                'Latency (s)': latency
            })

    # Ghi kết quả ra file CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'FPS', 'Latency (s)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\nResults saved to {OUTPUT_CSV}")
    print("=" * 50)
    for result in results:
        print(f"Model: {result['Model']}")
        print(f"FPS: {result['FPS']:.2f}")
        print(f"Latency: {result['Latency (s)']:.4f} s")
        print("-" * 50)

if __name__ == "__main__":
    main()
