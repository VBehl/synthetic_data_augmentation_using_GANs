import torch
import torch.nn as nn

print("🔍 DEBUG: You are running cnn_debug.py from the right file.")

class MyBottleNet(nn.Module):
    def _init_(self):  # ✅ must include 'self'
        print("✅ Inside MyBottleNet _init_")
        super()._init_()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ✅ Instantiate model
model = MyBottleNet()
print("✅ Model architecture:")
print(model)
params = list(model.parameters())
print("✅ Total trainable params:", sum(p.numel() for p in params if p.requires_grad))