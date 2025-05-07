import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# 1. Hyperparams
DATA_DIR    = "Dataset"
BATCH_SIZE  = 32
NUM_EPOCHS  = 5
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Transforms & DataLoaders
train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", train_tfms)
test_ds  = datasets.ImageFolder(f"{DATA_DIR}/test",  test_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# 3. Model: transfer-learn ResNet18
model = models.resnet18(pretrained=True)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model = model.to(DEVICE)

# 4. Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# 5. Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# 6. Evaluate on test set
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc*100:.2f}%")

# 7. Save model & class mapping
torch.save(model.state_dict(), "model.pth")
with open("class_idx.json", "w") as f:
    json.dump(train_ds.class_to_idx, f)
print("Saved model.pth and class_idx.json")
