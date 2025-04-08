
# 🔍 Synthetic Data Augmentation with CycleGAN Defect Detection with ResNet18

A computer vision pipeline for visual defect detection on the MVTec AD dataset using synthetic data generation via CycleGAN and classification via fine-tuned ResNet18.

---

## 🚀 Project Overview
This project enhances defect detection using a combination of synthetic data generation and deep learning classification:

- **CycleGAN** learns to translate "good" images into realistic defect images.
- **ResNet18** is fine-tuned to classify images as "good" or "defect" using both real and synthetic images.
- Emphasis is placed on creating **balanced training and testing datasets** to avoid overfitting and improve generalization.

---

## 🧠 Methodology
1. **Data Preparation**
   - Extracted and balanced real samples from MVTec AD categories (`bottle`, `cable`, etc.)
   - Created `trainA` (good images) and `trainB` (real defects) for CycleGAN training

2. **Synthetic Data Generation**
   - Trained **CycleGAN** using good ↔ defect samples
   - Generated realistic defect images from good samples to increase diversity

3. **ResNet18 Classification**
   - Used transfer learning (pretrained ResNet18)
   - Unfroze layers `layer3`, `layer4`, `fc` for fine-tuning
   - Trained using a mix of real + synthetic defect images

4. **Evaluation & Visualization**
   - Evaluated with metrics: **Accuracy, Precision, Recall, F1, AUC, MCC**
   - Visualized with **ROC Curve, Confusion Matrix, Grad-CAM**, and training graphs

---

## 📦 Features
- Balanced defect dataset creation
- CycleGAN-based defect generation
- ResNet18 classifier with transfer learning
- Early stopping and performance logging
- Evaluation: classification report, AUC, sensitivity/specificity
- Visualizations: confusion matrix, training loss/accuracy, Grad-CAM

---

## 📊 Results (Sample: Cable Category)
| Metric                | Value     |
|-----------------------|-----------|
| Accuracy              | 96.59%    |
| Precision             | 1.00      |
| Recall (Sensitivity)  | 0.93      |
| F1 Score              | 0.97      |
| ROC AUC               | 0.98      |

---

## 🛠️ Requirements
- Python 3.10
- PyTorch, torchvision
- scikit-learn
- matplotlib
- OpenCV


## 🖼️ Visual Outputs
- `confusion_matrix.png`
- `roc_curve.png`
- `training_loss_vs_epoch.png`
- `accuracy_vs_epoch.png`
- `GradCAM` visual heatmaps

---

## 📚 References
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [PyTorch ResNet18](https://pytorch.org/vision/stable/models.html)


---

## 📌 License
This project is for academic and research use only. Feel free to adapt with credit.


