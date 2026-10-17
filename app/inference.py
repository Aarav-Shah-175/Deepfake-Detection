"""
Inference engine for Flask Web Application.
Maintains a singleton PyTorch model in memory for fast real-time inference.
"""

import os
import sys
from PIL import Image

# Ensure model directory is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model.architecture import load_detector, predict_single_image

# Singleton model instance
_MODEL = None
_DEVICE = None

def get_detector():
    global _MODEL, _DEVICE
    if _MODEL is None:
        _MODEL, _DEVICE = load_detector()
    return _MODEL, _DEVICE

def predict_image(image_path):
    """
    Runs deepfake detection on a static image file.
    """
    model, device = get_detector()
    try:
        res = predict_single_image(model, image_path, device=device)
        return {
            "success": True,
            "type": "image",
            "prediction_label": res["label"],
            "confidence": f"{res['confidence']}%",
            "confidence_value": res["confidence"],
            "prob_real": res["prob_real"],
            "prob_fake": res["prob_fake"],
            "raw_logit": res["raw_logit"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def predict_video(video_path, max_frames=16):
    """
    Extracts frames from video, runs face/frame inference, and aggregates predictions.
    """
    try:
        import cv2
    except ImportError:
        return {
            "success": False,
            "error": "OpenCV (cv2) is not installed. Please install opencv-python to enable video inference."
        }

    model, device = get_detector()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    if total_frames <= 0:
        cap.release()
        return {"success": False, "error": "Invalid or unreadable video file."}

    # Setup Haar Cascade for face detection if available
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path) if os.path.exists(face_cascade_path) else None

    step = max(1, total_frames // max_frames)
    frame_predictions = []
    
    curr = 0
    while cap.isOpened() and len(frame_predictions) < max_frames and curr < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr)
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_img = None
        
        if face_cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                pad_w, pad_h = int(w * 0.1), int(h * 0.1)
                y1 = max(0, y - pad_h)
                y2 = min(frame.shape[0], y + h + pad_h)
                x1 = max(0, x - pad_w)
                x2 = min(frame.shape[1], x + w + pad_w)
                face_img = rgb[y1:y2, x1:x2]

        if face_img is None:
            # Center crop square
            h, w, _ = rgb.shape
            dim = min(h, w)
            y1 = (h - dim) // 2
            x1 = (w - dim) // 2
            face_img = rgb[y1:y1+dim, x1:x1+dim]

        try:
            res = predict_single_image(model, face_img, device=device)
            res["frame_index"] = curr
            res["timestamp"] = round(curr / fps, 2)
            frame_predictions.append(res)
        except Exception:
            pass

        curr += step

    cap.release()

    if not frame_predictions:
        return {"success": False, "prediction_label": "without face", "error": "No valid frames could be analyzed."}

    avg_real = sum(f["prob_real"] for f in frame_predictions) / len(frame_predictions)
    pred_label = "Real" if avg_real > 0.5 else "Fake"
    confidence = avg_real if pred_label == "Real" else (1.0 - avg_real)
    fake_count = sum(1 for f in frame_predictions if f["label"] == "Fake")
    real_count = len(frame_predictions) - fake_count

    return {
        "success": True,
        "type": "video",
        "prediction_label": pred_label,
        "confidence": f"{round(confidence * 100, 2)}%",
        "confidence_value": round(confidence * 100, 2),
        "prob_real": round(avg_real, 4),
        "prob_fake": round(1.0 - avg_real, 4),
        "frames_analyzed": len(frame_predictions),
        "fake_frames": fake_count,
        "real_frames": real_count,
        "breakdown": frame_predictions
    }
