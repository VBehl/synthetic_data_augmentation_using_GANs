import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Paths
data_dir = "data/classifier_data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Transforms (no augmentation here)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Class labels:", train_dataset.classes)

# Load pretrained ResNet18 and modify
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer3, layer4, and fc
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters:", total_params)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Early stopping
patience = 5
no_improve_epochs = 0
best_accuracy = 0.0
best_model_state = None

# === Logging lists ===
train_losses = []
test_accuracies = []

# === Training loop ===
epochs = 25
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    scheduler.step()
    print(f"Epoch [{epoch}/{epochs}] Loss: {epoch_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = model.state_dict()
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    print(f"\tTest Accuracy: {accuracy:.2f}%")

    if no_improve_epochs >= patience:
        print(f"\n⏹️ Early stopping at epoch {epoch} — no improvement for {patience} epochs.")
        break

# === Save the best model ===
print(f"\n✅ Best Test Accuracy: {best_accuracy:.2f}%")
if best_model_state is not None:
    model_path = os.path.join(os.getcwd(), "resnet18_best_model.pth")
    
    

# === Save training metrics ===
metrics_path = os.path.join(os.getcwd(), "training_metrics.json")
with open(metrics_path, "w") as f:
    json.dump({"loss": train_losses, "accuracy": test_accuracies}, f)

print(f"📊 Training metrics saved to: {metrics_path}")


