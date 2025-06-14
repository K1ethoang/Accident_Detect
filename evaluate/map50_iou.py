import os
import csv
import numpy as np
from ultralytics import YOLO
import torch
from tqdm import tqdm
import cv2
import yaml
import glob

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    box format: [x1, y1, x2, y2]
    """
    # Get the coordinates of bounding boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Get the coordinates of intersection rectangle
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    # Intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    
    # Union Area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def evaluate_model(model_path, test_data_path, conf_threshold=0.5, iou_threshold=0.5):
    # Load the model
    model = YOLO(model_path)
    
    # Run validation on test dataset
    results = model.val(data=test_data_path, 
                        conf=conf_threshold, 
                        iou=iou_threshold, 
                        verbose=True)
    
    # Extract mAP50 from results
    map50 = results.box.map50
    
    # Get average IoU from model predictions
    # We'll need to calculate this manually as it's not directly provided in the validation metrics
    all_ious = []
    
    # Load YAML
    with open(test_data_path, 'r') as f:
        data_yaml = yaml.safe_load(f)

    # Lấy thư mục gốc của file YAML
    yaml_base_dir = os.path.dirname(os.path.abspath(test_data_path))

    # Ghép đường dẫn ảnh test dựa vào giá trị 'val' trong YAML
    test_img_dir = os.path.normpath(os.path.join(yaml_base_dir, data_yaml['val']))

    # Tự động suy ra thư mục label (giả sử cấu trúc giống YOLO: images <-> labels)
    label_dir = test_img_dir.replace("images", "labels")

    # In debug
    # print(f"YAML base dir       : {yaml_base_dir}")
    # print(f"Resolved test_img_dir: {test_img_dir}")
    # print(f"Resolved label_dir   : {label_dir}")

    image_paths = None
    
   # Kiểm tra thư mục và các file trong thư mục
    if os.path.exists(test_img_dir):
        # print(f"Directory {test_img_dir} exists!")
        # print(f"Files in {test_img_dir}:")
        # print(os.listdir(test_img_dir))  # In danh sách file trong thư mục

        # Lấy tất cả ảnh (jpg, png, jpeg)
        image_paths = glob.glob(os.path.join(test_img_dir, "**", "*.jpg"), recursive=True)
        image_paths += glob.glob(os.path.join(test_img_dir, "**", "*.png"), recursive=True)
        image_paths += glob.glob(os.path.join(test_img_dir, "**", "*.jpeg"), recursive=True)

        # print(f'Found {len(image_paths)} image(s):')
        # print(f'image_paths {image_paths}')
    else:
        print(f"Directory {test_img_dir} does not exist!")
       
    for img_path in tqdm(image_paths, desc="Calculating Average IoU"):
        # Load image
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # Load label file
        label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        true_boxes = []
        if os.path.exists(label_path):
            # print(f'load label file')
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    # Convert to pixel coords
                    x1 = (x_center - w / 2) * width
                    y1 = (y_center - h / 2) * height
                    x2 = (x_center + w / 2) * width
                    y2 = (y_center + h / 2) * height
                    true_boxes.append([x1, y1, x2, y2])

        # Predict
        results = model.predict(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
        predictions = results[0].boxes

        pred_boxes = []
        if predictions is not None and predictions.xyxy is not None:
            for box in predictions.xyxy:
                pred_boxes.append(box[:4].tolist())
                
        # Calculate IoUs
        for pred_box in pred_boxes:
            ious = [calculate_iou(pred_box, true_box) for true_box in true_boxes]
            # print(f'ious {ious}')
            if ious:
                all_ious.append(max(ious))
    
    # Calculate average IoU
    avg_iou = np.mean(all_ious) if all_ious else 0
    print(f'avg_iou {avg_iou}')
    
    return {"mAP50": map50, "Average_IoU": avg_iou}

def main():
    # Thư mục chứa các mô hình YOLOv11 (v11_n, v11_s, v11_m, ...)
    model_root_dir = "./runs/train/"
    test_data_path = "./dataset/data_evaluate.yaml"
    
    output_csv = "./evaluate/mAP50_IoU.csv"
    results = []

    # Duyệt qua tất cả thư mục con trong model_root_dir
    for version_dir in os.listdir(model_root_dir):
        version_path = os.path.join(model_root_dir, version_dir)
        weights_dir = os.path.join(version_path, "weights")

        if not os.path.isdir(weights_dir):
            continue

        # Duyệt qua best.pt nếu tồn tại
        for weight_file in ["best.pt"]:
            model_path = os.path.join(weights_dir, weight_file)
            if os.path.isfile(model_path):
                print(f"Evaluating model: {model_path}")
                eval_result = evaluate_model(model_path, test_data_path)

                results.append({
                    "Model": "YOLOv11",
                    "Version": version_dir,
                    "Weight": weight_file,
                    "mAP50": eval_result["mAP50"],
                    "Average_IoU": eval_result["Average_IoU"]
                })

    # Ghi ra CSV theo đúng thứ tự n -> s -> m -> l -> x
    version_order = {"v11-n": 0, "v11-s": 1, "v11-m": 2, "v11-l": 3, "v11-x": 4}
    results.sort(key=lambda x: version_order.get(x["Version"], 999))

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ["Model", "Version", "Weight", "mAP50", "Average_IoU"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\n✅ All evaluation results saved to {output_csv}")

if __name__ == "__main__":
    main()
