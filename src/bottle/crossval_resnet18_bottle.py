# crossval_resnet18_bottle.py  – 5‑fold stratified CV + calibrated test run
# Place in src/bottle_src/ and run from project root:
#   python src/bottle_src/crossval_resnet18_bottle.py

import os
import json
import random
import numpy as np
import torch
import copy
import warnings
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
# from torchmetrics.classification import BinaryCalibration
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = "data/classifier_data/bottle"  # adjust if needed

# --------------- Helpers ---------------

def build_model():
    """Build a ResNet18 with partial fine-tuning (layer4 + fc)."""
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze everything
    for p in m.parameters():
        p.requires_grad = False
    # Unfreeze layer4 + fc
    for n, p in m.named_parameters():
        if any(k in n for k in ["layer4", "fc"]):
            p.requires_grad = True
    # Replace fc with Dropout + Linear
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, 2)
    )
    return m.to(DEVICE)

def train_one_epoch(model, loader, criterion, opt):
    """Train for 1 epoch on given loader."""
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def eval_loader(model, loader, criterion):
    """
    Evaluate on a loader, returning (avg_loss, true_labels, predicted_prob_of_class1).
    Uses softmax to get probability of class1.
    """
    model.eval()
    running_loss = 0.0
    y_true = []
    y_prob = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)

            # Probability of class1 (index=1)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            y_prob.extend(prob)
            y_true.extend(y.cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, np.array(y_true), np.array(y_prob)


# --------------- Transform & Full Dataset ---------------
TFM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# We'll do 5-fold CV on everything in 'train/' folder
full_ds = datasets.ImageFolder(os.path.join(BASE, "train"), transform=TFM)
X_idx = np.arange(len(full_ds))
y_full = np.array([lbl for _, lbl in full_ds.imgs])
CLASS_NAMES = full_ds.classes

# --------------- Cross-Validation ---------------
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_acc, fold_auc = [], []
print("\n📋 5‑fold cross‑validation on the training folder:")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_idx, y_full), 1):
    tr_ds = Subset(full_ds, tr_idx)
    va_ds = Subset(full_ds, va_idx)
    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=16, shuffle=False)

    # Compute per-fold class weights
    tr_labels = [y_full[i] for i in tr_idx]
    counts = np.bincount(tr_labels)
    w = torch.tensor(1.0 / counts, dtype=torch.float, device=DEVICE)
    w = w / w.sum() * len(counts)
    criterion = nn.CrossEntropyLoss(weight=w)

    model = build_model()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=5e-4, weight_decay=1e-4)
    sched = StepLR(opt, step_size=5, gamma=0.5)

    best_acc = 0.0
    best_state = None
    wait = 0
    PATIENCE = 5

    for epoch in range(1, 26):
        _ = train_one_epoch(model, tr_loader, criterion, opt)
        va_loss, va_true, va_prob = eval_loader(model, va_loader, criterion)
        va_acc = accuracy_score(va_true, va_prob > 0.5)

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        sched.step()

        if wait >= PATIENCE:
            # Early stop if no improvement in PATIENCE epochs
            break

    # Load best model of this fold
    model.load_state_dict(best_state)
    _, va_true, va_prob = eval_loader(model, va_loader, criterion)
    acc_fold = accuracy_score(va_true, va_prob > 0.5) * 100
    auc_fold = roc_auc_score(va_true, va_prob)
    fold_acc.append(acc_fold)
    fold_auc.append(auc_fold)
    print(f"  Fold {fold}: Acc {acc_fold:.2f}% | AUC {auc_fold:.3f}")

print(f"\nMean ± Std  ACC: {np.mean(fold_acc):.2f} ± {np.std(fold_acc):.2f}%")
print(f"Mean ± Std  AUC: {np.mean(fold_auc):.3f} ± {np.std(fold_auc):.3f}")

# --------------- Retrain on train+val, then test ---------------
print("\n🏁  Retrain on train+val folders, then evaluate test folder...")

# Build combined dataset from train/ + val/
train_ds = datasets.ImageFolder(os.path.join(BASE, "train"), transform=TFM)
val_ds   = datasets.ImageFolder(os.path.join(BASE, "val"),   transform=TFM)
comb_ds  = ConcatDataset([train_ds, val_ds])
comb_loader = DataLoader(comb_ds, batch_size=16, shuffle=True)

# Weighted loss for combined set
labels_comb = [s[1] for s in train_ds.samples] + [s[1] for s in val_ds.samples]
counts = np.bincount(labels_comb)
w = torch.tensor(1.0 / counts, dtype=torch.float, device=DEVICE)
w = w / w.sum() * len(counts)
criterion = nn.CrossEntropyLoss(weight=w)

model = build_model()
opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=5e-4, weight_decay=1e-4)
sched = StepLR(opt, step_size=5, gamma=0.5)

best_acc, best_state = 0.0, None
wait, PATIENCE = 0, 5

for epoch in range(1, 26):
    train_one_epoch(model, comb_loader, criterion, opt)
    sched.step()

    # Quick internal val approach: random 10% of comb for early-stop
    model.eval()
    sample_idx = random.sample(range(len(comb_ds)), k=int(0.1*len(comb_ds)))
    xs = []
    ys = []
    for idx in sample_idx:
        x_img, y_lbl = comb_ds[idx]
        xs.append(x_img)
        ys.append(y_lbl)
    xs_t = torch.stack(xs).to(DEVICE)
    ys_t = torch.tensor(ys).to(DEVICE)
    with torch.no_grad():
        preds = model(xs_t).argmax(1)
        val_acc = (preds == ys_t).float().mean().item()

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    if wait >= PATIENCE:
        break

# Load best state
model.load_state_dict(best_state)

# --- Temperature calibration on val/ ---
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
val_logits = []
val_labels = []
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        logits = model(x).cpu()
        val_logits.append(logits)
        val_labels.extend(y.numpy())
val_logits = torch.cat(val_logits, dim=0)
val_labels = np.array(val_labels)

# calibrator = BinaryCalibration(method="temperature")
# calibrator.fit(val_logits, torch.tensor(val_labels))

# --- Final test evaluation ---
test_set = datasets.ImageFolder(os.path.join(BASE, "test"), transform=TFM)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

y_true, y_prob = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x).cpu()
        # Use calibrated probabilities
        # probs_calib = calibrator.transform(logits)[:, 1]
        probs = torch.softmax(logits,1)[:, 1].numpy()
        y_prob.extend(probs)
        y_true.extend(y.numpy())
        # y_prob.extend(probs_calib.numpy())

y_pred = (np.array(y_prob) > 0.5).astype(int)
acc_test = accuracy_score(y_true, y_pred) * 100
auc_test = roc_auc_score(y_true, y_prob)
print(f"\n✅  Final TEST accuracy: {acc_test:.2f}%  (best val {best_acc*100:.2f}%)\n")
print("Classification Report (test):")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Bottle Confusion Matrix')
os.makedirs('results', exist_ok=True)
plt.tight_layout()
plt.savefig('results/bottle_confusion.png')
plt.show()

# --- Summaries ---
cv_summary = {
    "fold_acc": list(map(float, fold_acc)),
    "fold_auc": list(map(float, fold_auc)),
    "mean_acc": float(np.mean(fold_acc)),
    "std_acc": float(np.std(fold_acc)),
    "mean_auc": float(np.mean(fold_auc)),
    "std_auc": float(np.std(fold_auc)),
    "final_test_acc": float(acc_test),
    "final_test_auc": float(auc_test)
}

os.makedirs('results', exist_ok=True)
with open('results/bottle_cv_summary.json','w') as f:
    json.dump(cv_summary, f, indent=2)

# Save final model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/resnet18_bottle_best.pth")
print("\nArtifacts saved to models/ and results/.\n")