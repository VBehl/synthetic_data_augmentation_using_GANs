print("✅ File is executing...")

import torch
import torch.nn as nn

class TestModel(nn.Module):
    def _init_(self):  # ✅ NOTE: MUST include self
        print("✅ Inside TestModel _init_")
        super()._init_()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

# ✅ Instantiate the model
model = TestModel()
print("✅ Model object:", model)

# ✅ Check parameter count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("✅ Trainable parameters:", total_params)