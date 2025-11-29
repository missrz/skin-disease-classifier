import io
import json
import torch
import numpy as np
import cv2
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from bson.objectid import ObjectId
import os
from datetime import datetime
from jose import jwt

# Local app imports for auth
from database import db
from models.user import UserCreate, UserLogin, UserResponse, UserDB
from auth import hash_password, verify_password, create_access_token

# 1. Load model + class mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    with open("class_idx.json") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
except Exception as e:
    raise RuntimeError("Failed to load class mapping.") from e

try:
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device).eval()
except Exception as e:
    raise RuntimeError("Model initialization failed.") from e

# 2. Preprocess pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. FastAPI app
app = FastAPI(title="Skin Disease Classifier")

# 3a. Prediction schema
class SinglePrediction(BaseModel):
    disease: str
    confidence: float

class Prediction(BaseModel):
    top_predictions: List[SinglePrediction]
    warning: Optional[str] = None

# 3b. Skin detection heuristic
def contains_skin(img: np.ndarray, threshold: float = 0.02) -> bool:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    mask_ycrcb = ((cr > 133) & (cr < 173) & (cb > 77) & (cb < 127))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_hsv = ((h < 50) & (s > 58) & (s < 174) & (v > 50))

    mask = mask_ycrcb | mask_hsv
    skin_ratio = np.count_nonzero(mask) / mask.size
    return skin_ratio >= threshold

# 3c. /predict endpoint
MAX_IMG_SIZE_MB = 5

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...), check_skin: bool = True):
    data = await file.read()

    if len(data) > MAX_IMG_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large.")

    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_arr = np.array(pil_img)
    warning: Optional[str] = None

    if check_skin and not contains_skin(img_arr):
        warning = "Image does not contain detectable skin regions."

    x = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k=3)

    top_predictions = [
        {"disease": idx_to_class[int(i)], "confidence": float(c)}
        for c, i in zip(topk.values, topk.indices)
    ]

    return {"top_predictions": top_predictions, "warning": warning}

# 3d. Serve index.html at /
@app.get("/", response_class=HTMLResponse)
def read_index():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found.")

# 3e. Mount static assets
app.mount("/static", StaticFiles(directory="static"), name="static")


# -----------------------
# Auth endpoints
# -----------------------


@app.post("/signup", response_model=UserResponse)
async def signup(user: UserCreate):
    existing_user = await db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = hash_password(user.password)

    new_user = {
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_pw
    }

    result = await db.users.insert_one(new_user)

    return UserResponse(id=str(result.inserted_id), email=user.email, full_name=user.full_name)


@app.post("/login")
async def login(user: UserLogin):
    existing_user = await db.users.find_one({"email": user.email})
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(user.password, existing_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"user_id": str(existing_user["_id"])})

    return {"access_token": token, "token_type": "bearer"}


@app.post("/signout")
async def signout(token: str):
    # Simple token blacklist: store the token and expiry
    try:
        payload = jwt.decode(token, os.environ.get("SECRET_KEY", "replace-this-secret-in-production"), algorithms=[os.environ.get("ALGORITHM", "HS256")])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid token")

    exp = payload.get("exp")
    if not exp:
        raise HTTPException(status_code=400, detail="Token has no expiry")

    # Convert epoch exp to datetime for TTL indexing
    try:
        exp_dt = datetime.utcfromtimestamp(int(exp))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid token expiry")

    await db.token_blacklist.insert_one({"token": token, "exp": exp_dt})

    return {"detail": "Successfully signed out"}



@app.on_event("startup")
async def ensure_indexes():
    # Create TTL index on token_blacklist.exp so tokens are automatically removed after expiry
    try:
        await db.token_blacklist.create_index("exp", expireAfterSeconds=0)
    except Exception:
        # Index creation failure shouldn't stop the app; log if needed
        pass
