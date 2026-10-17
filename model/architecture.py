import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Class definitions
# 0 = Fake, 1 = Real
CLASS_NAMES = ["Fake", "Real"]

# Default ImageNet normalization used during training
IMAGE_SIZE = 224
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def build_mobilenet_model():
    """
    Constructs the MobileNetV3-Small deepfake detection model architecture.
    The final classifier layer is replaced with a single output logit for binary classification.
    """
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 1)
    return model

def get_default_weights_path():
    """Returns the default trained model checkpoint path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    primary = os.path.join(current_dir, "deepfake_mobilenet_20251110_162136.pth")
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(current_dir, "deepfake_mobilenet_acc_73.71.pth")
    if os.path.exists(fallback):
        return fallback
    return primary

def load_detector(weights_path=None, device=None):
    """
    Loads the trained MobileNetV3 deepfake detection model on the specified device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if weights_path is None:
        weights_path = get_default_weights_path()
        
    model = build_mobilenet_model().to(device)
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model weights not found at {weights_path}. Model initialized with random weights.")
        
    model.eval()
    return model, device

def predict_single_image(model, image_input, device="cpu", transform=None):
    """
    Predicts whether a single image is Fake or Real.
    image_input: PIL Image or file path (str)
    Returns: dict with label, confidence, score, and prob_real
    """
    if transform is None:
        transform = DEFAULT_TRANSFORM

    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        # assume numpy array
        img = Image.fromarray(image_input).convert("RGB")

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).item()
        prob_real = torch.sigmoid(torch.tensor(logit)).item()

    # In training, Fake=0, Real=1
    pred_idx = 1 if prob_real > 0.5 else 0
    label = CLASS_NAMES[pred_idx]
    confidence = prob_real if pred_idx == 1 else (1.0 - prob_real)

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "confidence_fraction": confidence,
        "prob_real": round(prob_real, 4),
        "prob_fake": round(1.0 - prob_real, 4),
        "raw_logit": round(logit, 4)
    }
