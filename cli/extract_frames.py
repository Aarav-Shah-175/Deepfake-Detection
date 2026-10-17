import cv2
import os
from glob import glob
from tqdm import tqdm

RAW_DIR = "data/raw"
CROPS_DIR = "data/crops"
FRAMES_PER_VIDEO = 50
FRAME_STEP = 10

def extract_frames(video_path, output_dir, max_frames=FRAMES_PER_VIDEO, step=FRAME_STEP):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % step == 0:
            frame_path = os.path.join(output_dir, f"{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
        frame_id += 1
    cap.release()

def main():
    for label in ["real", "fake"]:
        videos = glob(os.path.join(RAW_DIR, label, "*.mp4"))
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(CROPS_DIR, split, label), exist_ok=True)
        n_total = len(videos)
        n_val = max(1, int(0.1 * n_total))
        n_test = max(1, int(0.1 * n_total))
        n_train = n_total - n_val - n_test
        splits = {
            "train": videos[:n_train],
            "val": videos[n_train:n_train+n_val],
            "test": videos[n_train+n_val:]
        }
        for split, vids in splits.items():
            for video in tqdm(vids, desc=f"{split}/{label}"):
                vid_name = os.path.splitext(os.path.basename(video))[0]
                out_dir = os.path.join(CROPS_DIR, split, label, vid_name)
                extract_frames(video, out_dir)

if __name__ == "__main__":
    main()
