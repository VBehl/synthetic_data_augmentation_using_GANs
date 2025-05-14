import os, json, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = "data/classifier_data/cable"
CLASS_NAMES = ["defect", "good"]

# === Transform ===
tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_ds = datasets.ImageFolder(os.path.join(BASE, "train"), tfm)
test_ds  = datasets.ImageFolder(os.path.join(BASE, "test"), tfm)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)

# === Model ===
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if "layer4" in name or "fc" in name:
        p.requires_grad = True
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
model = model.to(DEVICE)

# === Loss, optimizer ===
labels = [y for _, y in train_ds]
w = torch.tensor(1 / np.bincount(labels), dtype=torch.float)
w = w / w.sum() * 2
criterion = nn.CrossEntropyLoss(weight=w.to(DEVICE))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)

# === Training ===
best_acc, best_state, wait = 0, None, 0
train_losses, val_accs = [], []
for epoch in range(1, 26):
    model.train(); run = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x); loss = criterion(out, y)
        loss.backward(); optimizer.step()
        run += loss.item() * x.size(0)
    train_loss = run / len(train_loader.dataset)

    # Eval
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total * 100

    print(f"Epoch {epoch:02d}  TrainLoss {train_loss:.4f}  TestAcc {acc:.2f}%")
    train_losses.append(train_loss); val_accs.append(acc)

    if acc > best_acc:
        best_acc, best_state, wait = acc, model.state_dict(), 0
    else:
        wait += 1
    if wait == 5: break

# === Save model and metrics ===
os.makedirs("models", exist_ok=True)
torch.save(best_state, "models/resnet18_cable_best.pth")
with open("results/cable_metrics.json", "w") as f:
    json.dump({"loss": train_losses, "accuracy": val_accs}, f)
print(f"\n✅  Best Test Accuracy: {best_acc:.2f}%")

# === Final Evaluation ===
model.load_state_dict(best_state)
model.eval(); y_true, y_prob = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x).cpu()
        prob = torch.softmax(logits, 1)[:, 1]
        y_true.extend(y.numpy()); y_prob.extend(prob.numpy())

y_pred = (np.array(y_prob) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Cable Confusion Matrix")
os.makedirs("results", exist_ok=True)
plt.tight_layout(); plt.savefig("results/cable_confusion.png"); plt.show()

fpr, tpr, _ = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)
plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve – Cable"); plt.legend(); plt.grid()
plt.savefig("results/cable_roc.png"); plt.show()