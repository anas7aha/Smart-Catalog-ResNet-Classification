# 👕 SmartCatalog: AI-Powered Apparel Classification App

An end-to-end, high-performance image classification pipeline designed for e-commerce platforms. This project implements a **Custom Deep Residual Network (ResNet) from scratch** (without transfer learning) using **PyTorch**, evaluated comprehensively using industry-standard metrics, and deployed via an interactive **Streamlit** web application.

Designed and developed for the **Smart Systems for Pattern Recognition** course.

---

## 🚀 Key Features

- **Modular Software Architecture**: Well-structured directory separation keeping data, model definition, training pipeline, and UI backend independent (Separation of Concerns).
- **Deep Learning From Scratch**: Zero pre-trained weights or transfer learning. The model learns entirely from randomly initialized weights.
- **Custom ResNet Architecture**: Implements custom Residual Blocks with identity mappings (**Skip Connections**) to solve the vanishing gradient problem.
- **Comprehensive Evaluation**: Auto-generates Accuracy, Class-wise Precision, Recall, F1-Score, and a visual Heatmap of the Confusion Matrix.
- **Real-time Inference Web UI**: A lightweight, responsive Streamlit web portal allowing real-time image uploads with instant confidence score predictions.
- **Unseen Test Set Benchmarking**: Built-in interactive validation directly on hidden test sets to prove model generalization during live demos.

---

## 📂 Project Directory Structure

```text
SmartCatalog/
│
├── models/
│   └── smart_catalog_resnet.pth       # Saved optimal model weights
│
├── src/
│   ├── __init__.py                     # Makes src a Python package
│   ├── dataset.py                      # Data loading, transforms & splits
│   ├── model_arch.py                   # Custom ResNet & Residual Block
│   └── train.py                        # Training Loop & performance evaluation
│
├── test_samples/                       # Extracted unseen test images for testing
│
├── app.py                              # Streamlit real-time interactive UI
└── README.md                           # This documentation file

```

## 🔬 Tech Stack & Frameworks

Language: Python 3.13

Deep Learning: PyTorch & Torchvision

Statistical Evaluation: Scikit-Learn

Web Frontend: Streamlit

Visualization: Matplotlib & Seaborn

## ⚙️ How to Run Locally

1. Training and Evaluation
   To fetch the dataset, train the custom network, evaluate performance metrics, and save the weights, run:

```bash
      python src/train.py
```

2. Launching the Web Application
   To run the interactive Streamlit application on your localhost, execute:

```bash
      streamlit run app.py
```
