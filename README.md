# ImprovedLeNet5 for CIFAR-10 Classification

This repository contains my implementation of an improved LeNet-5 model optimized for the CIFAR-10 dataset. The goal is to modernize the classic LeNet-5 architecture by removing dense layers and integrating advanced techniques to boost performance and generalization.

---

## 📁 Repository Structure

```plaintext
├── data/                    # Downloaded CIFAR-10 dataset
├── improved_lenet5.ipynb    # Jupyter Notebook with full experiment
├── improved_lenet5.py       # Standalone training script
├── best_model.pth           # Saved best model checkpoint
├── requirements.txt         # Python dependencies
└── README.md                # Project overview and instructions
```

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ImprovedLeNet5.git
   cd ImprovedLeNet5
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> **Note:** For a CPU-only setup, you can install PyTorch from the official CPU wheels:
> ```bash
> pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
> ```

## 📝 Usage

### 1. Jupyter Notebook

- Open and run `improved_lenet5.ipynb` in JupyterLab or Jupyter Notebook.
- Cells are organized into data preparation, model definition, training loop, and visualization.

### 2. Python Script

- To train from command line:
  ```bash
  python improved_lenet5.py
  ```
- The script will:
  1. Load CIFAR-10 with augmentation
  2. Build the CNN-only ImprovedLeNet5 model
  3. Train with OneCycleLR, label smoothing, and early stopping
  4. Save `best_model.pth` on validation loss improvement
  5. Plot training/validation curves and report final test accuracy

## 🔧 Hyperparameters & Techniques

- **Augmentation:** RandomCrop, HorizontalFlip, ColorJitter, RandomErasing
- **Architecture:** 3 conv blocks (32→64→128), BatchNorm, GELU, Dropout
- **Classifier:** 1×1 conv + Global Average Pooling (CNN-only)
- **Loss:** CrossEntropy with label smoothing (0.1)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** OneCycleLR (max_lr=1e-2)
- **Early Stopping:** Patience = 5 epochs

## 📊 Results

- **Best Validation Accuracy:** ~76.2%
- **Test Accuracy:** ~78.1%
- Loss and accuracy plots are available in the notebook and output figures.

## 📖 Future Work

- Experiment with residual connections or depthwise-separable convolutions
- Apply MixUp/CutMix augmentation for further gains
- Fine-tune hyperparameters with grid or random search

---

> **Author:** Bekir Berk YILDIRIM (Third-year Software Engineering Student)

---

Feel free to open issues or submit pull requests for improvements!

