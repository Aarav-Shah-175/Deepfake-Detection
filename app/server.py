import os
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Add parent directory to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.inference import predict_image, predict_video, get_detector

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/detection.html")
def detection():
    return render_template("detection.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/health", methods=["GET"])
def health():
    _, device = get_detector()
    return jsonify({
        "status": "healthy",
        "service": "DeepFake MobileNet Detection Engine",
        "device": device,
        "model": "MobileNetV3-Small (Binary Classification)",
        "test_accuracy": "85.04%",
        "auc_score": "0.9300"
    }), 200

@app.route("/", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    # Check for uploaded file in various expected form keys
    uploaded_file = None
    for key in ["video", "image", "file", "media"]:
        if key in request.files and request.files[key].filename:
            uploaded_file = request.files[key]
            break

    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"success": False, "error": "No file uploaded. Please upload a video or image."}), 400

    filename = uploaded_file.filename
    ext = os.path.splitext(filename)[1].lower()
    save_path = os.path.join(STORAGE_DIR, filename)
    uploaded_file.save(save_path)

    try:
        if ext in VIDEO_EXTENSIONS:
            result = predict_video(save_path)
        elif ext in IMAGE_EXTENSIONS:
            result = predict_image(save_path)
        else:
            return jsonify({
                "success": False,
                "error": f"Unsupported file extension '{ext}'. Supported formats: {list(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)}"
            }), 400

        print(f"Prediction for {filename}: {result.get('prediction_label')} ({result.get('confidence')})")
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    # Pre-load the model at server startup
    print("[*] Pre-loading MobileNetV3 deepfake detector...")
    get_detector()
    print("[*] Starting Deepfake Detection Web Server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
