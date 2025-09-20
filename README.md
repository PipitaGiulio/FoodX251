# FoodX-251 Classification Project

## 📌 Overview  
This project explores **food image classification** on the [FoodX-251 dataset](https://research.fb.com/foodx-251/) using two different approaches:  

1. **Supervised Learning** – A custom CNN trained directly on labeled data.  
2. **Self-Supervised Learning (SSL)** – A contrastive learning approach (SimCLR-style) followed by fine-tuning on downstream classification.  

The goal is to compare both methods, highlight trade-offs, and analyze how supervision impacts generalization on fine-grained food categories.


---

## 📊 Dataset  
- **Source:** FoodX-251 (≈160K images, 251 classes).  
- **Reduced Dataset:** ~130K images (80/10/10 train/val/test split using stratified sampling).  
- **Preprocessing:**  
  - Resize to fixed input size (224×224 for supervised, 128×128 for SSL pretext).  
  - Augmentations: random crop/flip, color jitter, Gaussian blur, normalization, and grayscale (for SSL).  

---

## 🧠 Models  

### 🔹 Supervised CNN  
- **Architecture:** 5 convolutional blocks + fully connected layers (<5M parameters).  
- **Optimizer:** SGD with momentum, weight decay, OneCycleLR scheduler.  
- **Loss:** Cross-entropy.  
- **Performance:**  
  - Test Accuracy: **44.6%**  
  - Precision: **45.9%**  
  - Recall: **42.8%**

### 🔹 Self-Supervised CNN (SimCLR-inspired)  
- **Pretext Task:** Contrastive learning with NT-Xent loss (batch size 512 via gradient accumulation).  
- **Downstream Task:** Fine-tuning head with two strategies:  
  - **Only fully connected layers** → Test Accuracy: ~26%  
  - **Progressive unfreezing of convolutional layers** → Test Accuracy: ~34%  

---

## 📈 Results & Insights  
- **Supervised Learning outperformed SSL** by ~10% accuracy.  
- SSL features were useful but not optimal for fine-grained classification.  
- Larger batch sizes, more training data, or alternative pretext tasks could improve SSL performance.  
- Both methods struggled with overfitting due to the complexity of 251 food categories.  

---

## ⚙️ Requirements  
- Python 3.8+  
- PyTorch  
- torchvision  
- NumPy, Matplotlib, etc.  

Install dependencies:  
```bash
pip install -r requirements.txt

