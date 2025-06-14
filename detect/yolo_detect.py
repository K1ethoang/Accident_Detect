import torch
import cv2
import time
import os
from glob import glob
from typing import Dict, List, Optional
from ultralytics import YOLO

# Global variables
CLASSES = ['accident', 'bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']


def load_model(model_path: str, model_type: str, device: torch.device) -> object:
    """
    Load either YOLOv8 or YOLOv11 model

    Args:
        model_path: Path to the model weights
        device: Device to run the model on

    Returns:
        Loaded model
    """
    print(f"Loading {model_type} model from {model_path} on {device}...")

    if model_type.lower() == 'yolov8' or model_type.lower() == 'yolov11':
        model = YOLO(model_path)
        # Set the device
        model.to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def boxes_intersect(box1: List[float], box2: List[float]) -> bool:
    """
    Check if two bounding boxes intersect

    Args:
        box1: First box in [x1, y1, x2, y2] format
        box2: Second box in [x1, y1, x2, y2] format

    Returns:
        True if boxes intersect, False otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Check for intersection
    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)


def detect_on_video(
    model: object,
    video_path: str,
    classes: List[str],
    device: torch.device,
    score_threshold: float = 0.2,
    output_path: Optional[str] = None
) -> None:
    """
    Run object detection on a video using either YOLOv8 or YOLOv9

    Args:
        model: Loaded YOLOv8 or YOLOv11 model
        model_type: Either 'yolov8' or 'yolov11'
        video_path: Path to input video
        classes: List of class names
        device: Device to run inference on
        score_threshold: Confidence threshold for detections
        output_path: Path to save output video (optional)
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:  # Print status every 10 frames
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            print(f"Processing frame {frame_count}/{total_frames} ({fps_processing:.2f} FPS)")

        results = model(frame, conf=score_threshold)
        detections = process_results(results, classes)

        # Extract detection information
        boxes = detections['boxes']  # format: [x1, y1, x2, y2]
        scores = detections['scores']
        labels = detections['labels']

        # Initialize object counter dictionary
        object_counts = {class_name: 0 for class_name in classes}

        # Find accident boxes
        accident_indices = [i for i, label in enumerate(labels) if classes[label] == 'accident']
        accident_boxes = [boxes[i] for i in accident_indices]

        # Update accident count
        object_counts["accident"] = len(accident_indices)

        # Draw accident boxes first (red)
        for i in accident_indices:
            box = boxes[i]
            score = scores[i]
            xmin, ymin, xmax, ymax = map(int, box)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(frame, f"accident: {score:.2f}", (xmin, ymin - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw other objects only if they intersect with any accident box
        if len(accident_boxes) > 0:
            for i in range(len(boxes)):
                # Skip if this is an accident box (already drawn)
                if i in accident_indices:
                    continue

                box = boxes[i]
                label_idx = labels[i]
                score = scores[i]
                class_name = classes[label_idx]

                # Check if this box intersects with any accident box
                intersects_accident = any(boxes_intersect(box, acc_box) for acc_box in accident_boxes)

                if intersects_accident:
                    xmin, ymin, xmax, ymax = map(int, box)

                    # Update object counter
                    object_counts[class_name] += 1

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name}: {score:.2f}", (xmin, ymin - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw object counts in the upper right corner
        margin = 10
        y_offset = margin
        for class_name, count in object_counts.items():
            if count > 0:  # Only display classes that are present
                text = f"{class_name}: {count}"
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                )
                # Draw background rectangle
                cv2.rectangle(
                    frame,
                    (int(width - text_width - margin*2 - 50), int(y_offset + 50)),
                    (int(width - margin - 50), int(y_offset + text_height + margin + 50)),
                    (255, 255, 255),
                    -1  # Fill rectangle
                )
                # Draw text
                cv2.putText(
                    frame,
                    text,
                    (int(width - text_width - margin*1.5 - 50), int(y_offset + text_height + 50)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )
                y_offset += text_height + margin*1.5

        # Write the frame
        if output_path:
            out.write(frame)

    # Release resources
    cap.release()
    if output_path:
        out.release()
        print(f"Detection video saved to {output_path}")


def process_results(results, classes: List[str]) -> Dict:
    """
    Process YOLO results into a standardized format

    Args:
        results: YOLO results
        classes: List of class names

    Returns:
        Dictionary with boxes, scores, and labels
    """
    # Initialize return values
    boxes = []
    scores = []
    labels = []

    # YOLOv8 returns results in a different format
    # Extract the first result (assumes batch size of 1)
    result = results[0]

    # Extract boxes (convert from xywh to xyxy if needed)
    if hasattr(result, 'boxes'):
        # Extract boxes (already in xyxy format)
        for box in result.boxes:
            # Get the box coordinates
            xyxy = box.xyxy.cpu().numpy()[0]
            boxes.append(xyxy)

            # Get the confidence score
            conf = box.conf.cpu().numpy()[0]
            scores.append(conf)

            # Get the class index
            cls_idx = int(box.cls.cpu().numpy()[0])
            # Map model's class index to our class list index
            # Assuming model classes align with our class list
            labels.append(cls_idx)

    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels
    }

def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Video input path
    input_video_path = "./evaluate/video_test.mp4"

    # Root directory chứa các folder train model (YOLOv11)
    weights_root = "./runs/train"

    # Duyệt tất cả đường dẫn tới file best.pt
    weight_paths = glob(os.path.join(weights_root, "**", "weights", "best.pt"), recursive=True)

    if not weight_paths:
        print("❌ Không tìm thấy bất kỳ file best.pt nào trong runs/train/")
        return

    for weight_path in sorted(weight_paths):
        # Lấy tên phiên bản từ thư mục cha (ví dụ: v11_n)
        version_name = os.path.basename(os.path.dirname(os.path.dirname(weight_path)))
        print(f"\n🔍 Đang xử lý model: {version_name} ({weight_path})")

        try:
            # Load model
            model = load_model(weight_path, 'yolov11', device)

            # Tạo thư mục output
            output_path = f"./detect/outputs/{version_name}_output.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Run detection
            detect_on_video(
                model=model,
                video_path=input_video_path,
                classes=CLASSES,
                device=device,
                output_path=output_path
            )
        except Exception as e:
            print(f"⚠️ Lỗi với model {version_name}: {e}")


if __name__ == "__main__":
    main()