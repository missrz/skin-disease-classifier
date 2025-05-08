import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score

# 1. Hyperparams
DATA_DIR   = "Dataset"
BATCH_SIZE = 64
NUM_EPOCHS = 15
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Transforms
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# 3. Datasets & Loaders
full_train = datasets.ImageFolder(f"{DATA_DIR}/train", train_tfms)
num_train  = int(0.8 * len(full_train))
num_val    = len(full_train) - num_train
train_ds, val_ds = random_split(full_train, [num_train, num_val])
test_ds  = datasets.ImageFolder(f"{DATA_DIR}/test", val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# 4. Model & head
model = models.resnet18(pretrained=True)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(full_train.classes))
model = model.to(DEVICE)

# 5. Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# 6. Training & validation loops
best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    # ––– Train –––
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out   = model(imgs)
        loss  = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # ––– Validate –––
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds  = torch.argmax(logits, dim=1).cpu().tolist()
            val_preds.extend(preds)
            val_labels.extend(labels.tolist())

    val_acc = accuracy_score(val_labels, val_preds)
    scheduler.step(val_acc)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS}  "
        f"Train Loss: {running_loss/len(train_loader):.4f}  "
        f"Val Acc: {val_acc*100:.2f}%"
    )

    # ––– Unfreeze last block after 5 epochs –––
    if epoch == 4:
        for p in model.layer4.parameters():
            p.requires_grad = True

# 7. Final test evaluation
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds  = torch.argmax(logits, dim=1).cpu().tolist()
        test_preds.extend(preds)
        test_labels.extend(labels.tolist())

test_acc = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# 8. Save
torch.save(model.state_dict(), "model.pth")
with open("class_idx.json", "w") as f:
    json.dump(full_train.class_to_idx, f)
print("Saved model and class mapping.")
