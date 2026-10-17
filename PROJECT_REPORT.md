# Deepfake Detection System: Architectural Audit & Integration Report

---

## 1. Executive Summary

This report documents the forensic evaluation, consolidation, and deployment preparation of the **Deepfake Detection System** located in the workspace `d:\Project- Academic\AI`.

Prior to consolidation, the workspace contained four fragmented folders representing different development phases of the project:
1. `Deepfake` (33,073 files, ~314.6 MB): The primary dataset and initial notebook experiments.
2. `Deepfake_Final` (16 files, ~10.1 MB): The latest and highest-performing PyTorch MobileNetV3 machine learning pipeline.
3. `DeepFake-Detection` (21 files, ~8.5 MB): An interactive web application branch based on an obsolete Keras architecture.
4. `deepfake-mobile` (31 files, ~18.1 MB): An experimental Android/mobile branch containing video preprocessing scripts.

### Outcome of the Audit
- **Definitive ML Project**: `Deepfake_Final` was determined to be the **proper and final core machine learning model**, achieving **85.04% Test Accuracy** and **0.9300 ROC-AUC** across 140,000+ images.
- **Unified Master Repository**: All viable components (the final model, the web interface, the video inference engine, test media, and the complete 33,000-image dataset) were consolidated into **`Deepfake_Unified`**.
- **Current State**: The legacy development directories were safely decommissioned and deleted, leaving a clean, self-contained, GitHub-ready project.

---

## 2. Forensic Analysis of Legacy Branches

```
                      Project Evolution Timeline
                      
 [Sept 2025]             [Oct 2025]             [Oct 2025]            [Nov 2025]
 deepfake-mobile  ──► DeepFake-Detection ──►     Deepfake     ──►  Deepfake_Final
 (Mobile Prototype,   (Flask Web App,        (Dataset Hub,        (Final PyTorch Model,
  Incomplete Android)  Keras LSTM/VGG)        Oct 18 Checkpoint)   85.04% Accuracy)
```

### Detailed Evaluation

1. **`Deepfake_Final` (The Definitive Core ML Pipeline)**
   - **Trained Model Weights**: `deepfake_mobilenet_20251110_162136.pth` (5.93 MB, saved November 10, 2025).
   - **Performance Benchmarks**:
     - Final Test Accuracy: **85.04%**
     - ROC AUC Score: **0.9300**
     - Area Under Precision-Recall Curve (PR AUC): **0.9269**
     - Validation Accuracy: **81.56%**
   - **Key Artifacts**: Full Google Colab notebook (`Deepfake_Final.ipynb`), complete markdown and PDF execution logs, loss/accuracy plots, confusion matrix, ROC curve, and validation video.
   - *Verdict*: Formed the core machine learning foundation of `Deepfake_Unified`.

2. **`Deepfake` (The Dataset and Working Training Hub)**
   - **Dataset Asset**: Contained 33,064 preprocessed facial crop images (`Test/Fake`: 5,492, `Test/Real`: 5,413, `Train/Fake`: 22,159).
   - **Intermediate Weights**: `deepfake_mobilenet_acc_73.71_20251018_181746.pth` (early checkpoint from October 18, 2025).
   - *Action Taken*: The entire `Dataset/` directory was physically moved into `Deepfake_Unified/Dataset`, and the checkpoint was preserved for version history.

3. **`DeepFake-Detection` (Interactive Web Portal)**
   - Contained a Flask web interface with responsive styling (`style.css`), logos, templates (`index.html`, `detection.html`), and sample fake videos (`11.mp4`, `Fac123.mp4`).
   - The backend was originally tied to a non-existent Keras model (`vgg.h5`).
   - *Action Taken*: Rescued the web frontend, modernized its templates to support both image and video uploads, and rewired the backend to load the trained PyTorch MobileNetV3 engine.

4. **`deepfake-mobile` (Mobile Deployment Prototype)**
   - Contained incomplete/empty Android files and 0-byte training stubs, but possessed valuable utilities for video frame extraction (`extract_frames.py`), MTCNN face cropping (`preprocess.py`), and TorchScript mobile export (`export_mobile.py`).
   - *Action Taken*: Migrated the utility scripts into `Deepfake_Unified/cli/`.

---

## 3. The Consolidated Repository: `Deepfake_Unified`

### File & Directory Inventory

```
Deepfake_Unified/
├── README.md                      # Complete GitHub repository guide & badges
├── PROJECT_REPORT.md              # This architectural & audit report
├── LICENSE                        # MIT License
├── requirements.txt               # Validated dependencies
├── .gitignore                     # Git rules for Python, Flask, PyTorch & Jupyter
│
├── model/                         # Neural Network Architecture & Weights
│   ├── architecture.py            # Clean MobileNetV3-Small model builder & inference methods
│   ├── deepfake_mobilenet_20251110_162136.pth   # Final trained weights (Nov 10, 2025)
│   └── deepfake_mobilenet_acc_73.71.pth         # Secondary reference checkpoint (Oct 18, 2025)
│
├── app/                           # Interactive Web Portal (Flask)
│   ├── server.py                  # HTTP server & REST API endpoints
│   ├── inference.py               # Singleton PyTorch inference worker for media uploads
│   ├── static/                    # CSS stylesheets, logos, and graphics
│   │   ├── style.css
│   │   ├── face.jpg
│   │   ├── main.png
│   │   └── detection.png
│   ├── templates/                 # Web templates
│   │   ├── index.html             # Landing page
│   │   ├── detection.html         # Media upload and detection portal
│   │   └── about.html             # Documentation & metrics page
│   └── storage/                   # Runtime upload buffer (.gitkeep)
│
├── cli/                           # Command-Line Tools
│   ├── predict.py                 # Multi-format CLI predictor (image/video/dir)
│   ├── extract_frames.py          # Video frame extraction utility
│   ├── preprocess.py              # MTCNN face cropping utility
│   └── export_mobile.py           # TorchScript / Mobile export script
│
├── notebooks/                     # Academic Documentation & Training Notebooks
│   ├── Deepfake_Final.ipynb       # Definitive Colab / Jupyter training notebook
│   ├── Deepfake_Final.ipynb - Colab.pdf # Complete PDF execution document
│   ├── Deepfake_Final.md          # Markdown export with complete training logs
│   └── evaluation_plots/          # Metric plots
│       ├── output_9_0.png         # Loss and accuracy progression
│       ├── output_11_0.png        # Confusion Matrix
│       └── output_12_0.png        # ROC Curve (AUC 0.93) & PR Curve (AUC 0.9269)
│
├── samples/                       # Test Assets (Immediately runnable)
│   ├── fake_331.jpg
│   ├── fake_365.jpg
│   ├── fake_7.jpg
│   ├── real_11.jpg
│   └── test_videos/
│       ├── 11.mp4                 # Sample fake video
│       ├── Fac123.mp4             # Sample fake video
│       └── WhatsApp_Video_Demo.mp4# Sample demonstration video
│
└── Dataset/                       # Local training/testing dataset (33,000+ images)
```

---

## 4. How to Download, Install, and Run

### Step 1: Clone or Download the Repository

```bash
# Clone the repository
git clone https://github.com/Aarav-Shah-175/Deepfake-Detection.git
cd Deepfake-Detection
```
*(If running locally from the current workspace, navigate to `d:\Project- Academic\AI\Deepfake_Unified`)*.

---

### Step 2: Environment Setup

Create an isolated Python 3.10+ virtual environment and install dependencies:

```bash
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 3: Running the Interactive Web Portal

Start the Flask server:

```bash
python app/server.py
```

1. Open your web browser at: **`http://localhost:5000`**
2. Navigate to the **Detection** tab (`http://localhost:5000/detection.html`).
3. Select an image (`.jpg`, `.png`) or video (`.mp4`, `.avi`, `.mov`) from the `samples/` directory.
4. Click **"Analyze File"**. The server will process the media, extract faces, run inference through the MobileNetV3 model, and display:
   - **Verdict**: `REAL` or `FAKE`
   - **Confidence Percentage**: e.g., `65.26%`
   - **Detailed Breakdown**: Frames analyzed, fake vs. real frame counts, and raw probability metrics.

---

### Step 4: Running the Command-Line Interface (CLI)

#### 1. Single Image Evaluation:
```bash
python cli/predict.py --image samples/fake_365.jpg
```
*Output:*
```text
Result     : Fake
Confidence : 63.26%
Prob(Real) : 0.3674
Prob(Fake) : 0.6326
Raw Logit  : -0.5436
```

#### 2. Directory Batch Evaluation:
```bash
python cli/predict.py --dir samples
```
*Output:*
```text
Filename                       | Prediction | Confidence | Prob(Real)
-----------------------------------------------------------------
fake_331.jpg                   | Real       |  50.60%    | 0.5060
fake_365.jpg                   | Fake       |  63.26%    | 0.3674
fake_7.jpg                     | Fake       |  55.36%    | 0.4464
real_11.jpg                    | Real       |  50.58%    | 0.5058
```

#### 3. Video Stream Evaluation:
```bash
python cli/predict.py --video samples/test_videos/Fac123.mp4
```
*Output:*
```text
[*] Analyzing video: Fac123.mp4 (322 frames, 25.0 FPS)
==================================================
  Video          : Fac123.mp4
  Final Decision : FAKE
  Confidence     : 65.26%
  Frames Analyzed: 16 (Fake: 16, Real: 0)
  Avg Prob(Fake) : 0.6526
==================================================
```

---

### Step 5: Jupyter / Colab Training Notebook

To review or modify the model training pipeline:
```bash
jupyter notebook notebooks/Deepfake_Final.ipynb
```

---

## 5. GitHub Readiness Checklist

- [x] **Single Source of Truth**: All 4 legacy folders evaluated, merged, and deleted.
- [x] **Model Weights Included**: 5.93 MB trained checkpoint is under GitHub's 100 MB per-file limit and tracked directly in Git.
- [x] **Sample Test Media**: Both images and video samples are included for immediate verification upon cloning.
- [x] **Complete `.gitignore`**: Excludes `__pycache__`, virtual environments, IDE configurations, and temporary storage uploads.
- [x] **Open-Source License**: Standard MIT License file included.
- [x] **Clean Documentation**: High-quality `README.md` with badges, architecture diagrams, and quickstart instructions.
