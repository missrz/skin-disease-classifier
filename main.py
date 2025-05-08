import io
import json
import torch
import numpy as np
import cv2
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# 1. Load model + class mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("class_idx.json") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Initialize and load the pretrained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device).eval()

# 2. Preprocess pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. FastAPI app
app = FastAPI(title="Skin Disease Classifier")

# 3a. Prediction schema with optional warning
class Prediction(BaseModel):
    disease: str
    confidence: float
    warning: Optional[str] = None

# 3b. Skin detection heuristic (YCrCb + HSV) with threshold
def contains_skin(img: np.ndarray, threshold: float = 0.02) -> bool:
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # YCrCb mask
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    mask_ycrcb = ((cr > 133) & (cr < 173) & (cb > 77) & (cb < 127))
    # HSV mask
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_hsv = ((h < 50) & (s > 58) & (s < 174) & (v > 50))
    # Combine masks
    mask = mask_ycrcb | mask_hsv
    skin_ratio = np.count_nonzero(mask) / mask.size
    return skin_ratio >= threshold

# 3c. /predict endpoint
@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_arr = np.array(pil_img)
    warning: Optional[str] = None
    # Check for skin presence
    if not contains_skin(img_arr):
        warning = "Image does not contain detectable skin regions."

    # Preprocess and predict
    x = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    disease = idx_to_class[int(idx)]
    confidence = float(conf)

    return {"disease": disease, "confidence": confidence, "warning": warning}

# 3d. Serve index.html at /
@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# 3e. Mount static assets under /static
app.mount("/static", StaticFiles(directory="static"), name="static")
