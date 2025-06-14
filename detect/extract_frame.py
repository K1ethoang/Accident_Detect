import cv2
import os
import re

def extract_frames(video_path, output_folder, frames_per_second):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / frames_per_second))

    os.makedirs(output_folder, exist_ok=True)

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            filename = os.path.join(output_folder, f"{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"✅ Đã trích xuất {saved_count} ảnh từ '{os.path.basename(video_path)}' vào '{output_folder}'.")


def process_videos_in_folder(video_folder, output_root, frames_per_second):
    # name = 'video_test.mp4'
    # video_path = os.path.join(video_folder, name)
    # output_folder = os.path.join(output_root, name)
    # extract_frames(video_path, output_folder, frames_per_second)
    # return

    pattern = r"^v11-([a-zA-Z0-9]+)_output\.mp4$"

    for filename in os.listdir(video_folder):
        match = re.match(pattern, filename)
        if match:
            version = match.group(1)
            video_path = os.path.join(video_folder, filename)
            output_folder = os.path.join(output_root, f"v11-{version}")
            extract_frames(video_path, output_folder, frames_per_second)
        else:
            print(f"⏩ Bỏ qua file không đúng định dạng: {filename}")


# Ví dụ sử dụng
video_folder = "./evaluate"
output_root = "./detect/extract"
frames_per_second = 10

process_videos_in_folder(video_folder, output_root, frames_per_second)
