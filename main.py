import io
import json
import torch
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 1. Load model + class mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("class_idx.json") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device).eval()

# 2. Preprocess pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 3. FastAPI app
app = FastAPI(title="Skin Disease Classifier")

# 3a. Prediction schema
class Prediction(BaseModel):
    disease: str
    confidence: float

# 3b. /predict endpoint
@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    return {
        "disease": idx_to_class[int(idx)],
        "confidence": float(conf)
    }

# 3c. Serve index.html at /
@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# 3d. Mount the rest of your static assets under /static
app.mount("/static", StaticFiles(directory="static"), name="static")
