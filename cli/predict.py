"""
CLI Inference Tool for Deepfake Detection
Supports both individual images and videos.

Usage:
    python predict.py --image samples/fake_365.jpg
    python predict.py --video samples/test_videos/Fac123.mp4
    python predict.py --dir samples/
"""

import os
import sys
import argparse
import glob

# Add parent directory to path so model package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.architecture import load_detector, predict_single_image

def analyze_video(model, video_path, device="cpu", max_frames=20, step=10):
    """
    Extracts frames from video and analyzes face/frame authenticity.
    """
    try:
        import cv2
    except ImportError:
        print("[!] opencv-python is not installed. Video analysis requires OpenCV.")
        return None

    if not os.path.exists(video_path):
        print(f"[!] Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    if total_frames <= 0:
        print(f"[!] Could not read video frames from {video_path}")
        cap.release()
        return None

    print(f"[*] Analyzing video: {os.path.basename(video_path)} ({total_frames} frames, {fps:.1f} FPS)")
    
    # Use Haar Cascade face detector if available, otherwise analyze center-cropped frames
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path) if os.path.exists(face_cascade_path) else None

    frame_interval = max(1, total_frames // max_frames)
    extracted = 0
    frame_results = []
    
    current_frame = 0
    while cap.isOpened() and extracted < max_frames and current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face if cascade available
        face_img = None
        if face_cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                # Take largest face
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                # Expand box slightly
                pad_w, pad_h = int(w * 0.1), int(h * 0.1)
                y1 = max(0, y - pad_h)
                y2 = min(frame.shape[0], y + h + pad_h)
                x1 = max(0, x - pad_w)
                x2 = min(frame.shape[1], x + w + pad_w)
                face_img = rgb_frame[y1:y2, x1:x2]

        if face_img is None:
            # Fallback to center-crop square
            h, w, _ = rgb_frame.shape
            size = min(h, w)
            y1 = (h - size) // 2
            x1 = (w - size) // 2
            face_img = rgb_frame[y1:y1+size, x1:x1+size]

        res = predict_single_image(model, face_img, device=device)
        res["frame_idx"] = current_frame
        res["time_sec"] = round(current_frame / fps, 2)
        frame_results.append(res)

        extracted += 1
        current_frame += frame_interval

    cap.release()

    if not frame_results:
        return None

    # Aggregate results
    avg_prob_real = sum(r["prob_real"] for r in frame_results) / len(frame_results)
    pred_label = "Real" if avg_prob_real > 0.5 else "Fake"
    confidence = avg_prob_real if pred_label == "Real" else (1.0 - avg_prob_real)
    fake_frames_count = sum(1 for r in frame_results if r["label"] == "Fake")

    return {
        "video": os.path.basename(video_path),
        "overall_label": pred_label,
        "overall_confidence": round(confidence * 100, 2),
        "avg_prob_real": round(avg_prob_real, 4),
        "avg_prob_fake": round(1.0 - avg_prob_real, 4),
        "frames_analyzed": len(frame_results),
        "fake_frames": fake_frames_count,
        "real_frames": len(frame_results) - fake_frames_count,
        "frame_breakdown": frame_results
    }

def main():
    parser = argparse.ArgumentParser(description="DeepFake MobileNet Detector CLI")
    parser.add_argument("--image", type=str, help="Path to single image to classify")
    parser.add_argument("--video", type=str, help="Path to video file to classify")
    parser.add_argument("--dir", type=str, help="Path to folder of images to classify")
    parser.add_argument("--weights", type=str, default=None, help="Optional custom .pth weights path")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to extract from video")

    args = parser.parse_args()

    if not any([args.image, args.video, args.dir]):
        parser.print_help()
        sys.exit(1)

    print("=" * 60)
    print("      Deepfake MobileNet Detection Engine")
    print("=" * 60)

    model, device = load_detector(weights_path=args.weights, device=args.device)
    print(f"[*] Loaded model on device: {device.upper()}")

    if args.image:
        print(f"\n[*] Processing image: {args.image}")
        res = predict_single_image(model, args.image, device=device)
        print("-" * 50)
        print(f"  Result     : {res['label']}")
        print(f"  Confidence : {res['confidence']}%")
        print(f"  Prob(Real) : {res['prob_real']}")
        print(f"  Prob(Fake) : {res['prob_fake']}")
        print(f"  Raw Logit  : {res['raw_logit']}")
        print("-" * 50)

    if args.video:
        res = analyze_video(model, args.video, device=device, max_frames=args.frames)
        if res:
            print("\n" + "=" * 50)
            print(f"  Video          : {res['video']}")
            print(f"  Final Decision : {res['overall_label'].upper()}")
            print(f"  Confidence     : {res['overall_confidence']}%")
            print(f"  Frames Analyzed: {res['frames_analyzed']} (Fake: {res['fake_frames']}, Real: {res['real_frames']})")
            print(f"  Avg Prob(Fake) : {res['avg_prob_fake']}")
            print("=" * 50)

    if args.dir:
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_files = []
        for p in patterns:
            image_files.extend(glob.glob(os.path.join(args.dir, p)))
        
        print(f"\n[*] Found {len(image_files)} images in {args.dir}")
        print("-" * 65)
        print(f"{'Filename':<30} | {'Prediction':<10} | {'Confidence':<10} | {'Prob(Real)'}")
        print("-" * 65)
        for img_path in sorted(image_files):
            r = predict_single_image(model, img_path, device=device)
            fname = os.path.basename(img_path)
            print(f"{fname:<30} | {r['label']:<10} | {r['confidence']:>6.2f}%    | {r['prob_real']:>6.4f}")
        print("-" * 65)

if __name__ == "__main__":
    main()
