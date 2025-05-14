import os, json, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ==== DEVICE ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==== PATHS ====
BASE = "data/classifier_data/bottle"
TRAIN_DIR = os.path.join(BASE, "train")
VAL_DIR   = os.path.join(BASE, "val")
TEST_DIR  = os.path.join(BASE, "test")

# ==== TRANSFORMS ====
TFM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_ds = datasets.ImageFolder(TRAIN_DIR, TFM)
val_ds   = datasets.ImageFolder(VAL_DIR,   TFM)
test_ds  = datasets.ImageFolder(TEST_DIR,  TFM)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

CLASS_NAMES = train_ds.classes
print("Classes:", CLASS_NAMES)

# ==== CLASS WEIGHTS (from TRAIN) ====
train_targets = [lbl for _,lbl in train_ds.imgs]
class_counts  = Counter(train_targets)
weights = torch.tensor([1.0 / class_counts[c] for c in range(len(class_counts))],
                       dtype=torch.float, device=DEVICE)
weights = weights / weights.sum() * len(class_counts)
print("Class weights:", weights.cpu().numpy())

# ==== MODEL ====
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if any(k in name for k in ["layer4", "fc"]):
        p.requires_grad = True
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)
model.to(DEVICE)
print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ==== OPTIMISER / LOSS / SCHEDULER ====
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=5e-4, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ==== TRAIN LOOP WITH VAL EARLY‑STOP ====
EPOCHS, PATIENCE = 25, 5
wait, best_val = 0, 0.0
train_losses, val_losses, val_accs = [], [], []

for epoch in range(1, EPOCHS+1):
    # ---- train ----
    model.train(); running=0.0
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x); loss = criterion(out,y)
        loss.backward(); optimizer.step()
        running += loss.item()*x.size(0)
    train_loss = running/len(train_ds)
    train_losses.append(train_loss)
    scheduler.step()

    # ---- validation ----
    model.eval(); v_run, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); loss = criterion(out,y)
            v_run += loss.item()*x.size(0)
            v_total += y.size(0)
            v_correct += (out.argmax(1)==y).sum().item()
    val_loss = v_run/len(val_ds)
    val_acc  = 100*v_correct/v_total
    val_losses.append(val_loss); val_accs.append(val_acc)

    print(f"Epoch {epoch}/{EPOCHS}  TrainLoss {train_loss:.4f}  ValLoss {val_loss:.4f}  ValAcc {val_acc:.2f}%")

    if val_acc>best_val:
        best_val, best_state = val_acc, model.state_dict(); wait=0
    else:
        wait+=1
    if wait>=PATIENCE:
        print("Early‑stopping (no val improvement)."); break

# ==== LOAD BEST & TEST EVALUATION ====
model.load_state_dict(best_state)
model.eval(); t_correct, t_total = 0,0; y_true,y_pred=[],[]
with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        preds = out.argmax(1)
        t_total += y.size(0); t_correct += (preds==y).sum().item()
        y_true.extend(y.cpu().numpy()); y_pred.extend(preds.cpu().numpy())

test_acc = 100*t_correct/t_total
print(f"\n✅  Final TEST accuracy: {test_acc:.2f}%  (best val {best_val:.2f}%)")

# ---- classification report & confusion matrix ----
print("\nClassification Report (test):")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Bottle Confusion Matrix')
plt.tight_layout(); os.makedirs('results',exist_ok=True); plt.savefig('results/bottle_confusion.png'); plt.show()

# ==== SAVE ====  
os.makedirs('models',exist_ok=True)
torch.save(best_state, 'models/resnet18_bottle_best.pth')
with open('results/bottle_training_metrics.json','w') as f:
    json.dump({"train_loss":train_losses, "val_loss":val_losses, "val_acc":val_accs, "test_acc":test_acc}, f)
print("Artifacts saved to models/ and results/.")
