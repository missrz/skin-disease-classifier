# Skin Disease Classifier (FastAPI)

This repository contains a FastAPI service for classifying skin disease images using a pretrained PyTorch model. You can run it in two ways:

**Local Setup**: Python virtual environment + Uvicorn

---

## 📂 Repository Structure

```text
.
├── Dataset/            # Your train/test image folders
├── class_idx.json      # Generated class-to-index mapping
├── main.py             # FastAPI application (includes index route at "/")
├── model.pth           # Pretrained PyTorch model
├── requirements.txt    # Python dependencies
├── train.py            # Script to train and save the model
└── README.md           # This file
```

---

## ⚙️ Local Setup

### Prerequisites

* Python 3.11+
* `git`, `pip`

### 1. Clone Repository

```bash
git clone https://github.com/missrz/skin-disease-classifier.git
cd skin-disease-classifier
```

### 2. Create & Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare/Train the Model

If `model.pth` and `class_idx.json` already exist, skip this step.
Otherwise:

```bash
python train.py
```

This script will:

* Load images from `Dataset/train` and `Dataset/test`
* Fine-tune a ResNet18 model
* Save `model.pth` and `class_idx.json`

### 5. Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Test the Home Index

You can visit [http://localhost:8000/](http://localhost:8000/) in your browser to see the HTML-based home page (e.g., your index view).

Or use `curl` to fetch the raw HTML:

```bash
curl http://localhost:8000/
```

### 7. Test Prediction Endpoint

```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

---

## 📄 License

MIT ©ACET
