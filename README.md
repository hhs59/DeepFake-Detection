# DeepFake-Detection
# Real-time Deepfake Detection using CNN and PyTorch

## 📝 Overview
This project aims to develop a robust and efficient Deepfake detection system using Convolutional Neural Networks (CNN). The system focuses on real-time inference and high-precision classification between authentic and manipulated facial content.

The model is trained and validated on the **FaceForensics++** dataset, utilizing advanced facial landmarking for precise feature extraction.

---

## ✨ Key Features
- **Hardware Acceleration:** Fully optimized using PyTorch's MPS (Metal Performance Shaders) for high-speed training and inference on M2 Macs.
- **Advanced Preprocessing:** Real-time face detection and cropping using **MediaPipe**.
- **Hyperparameter Optimization:** Automated tuning of learning rates, architectures, and optimizers using **Optuna** (Bayesian Optimization).
- **Real-time Inference:** A dedicated module for live deepfake detection via webcam.

---

## 🛠 Tech Stack
- **Framework:** PyTorch (Core Deep Learning)
- **Computer Vision:** OpenCV, MediaPipe
- **Optimization:** Optuna
- **Development:** Python 3.13, VS Code, Git

---

## 📂 Project Structure
```text
├── data/               # (Hidden) Dataset and processed face crops
├── models/             # Saved model weights (.pth files)
├── notebooks/          # Exploratory Data Analysis (EDA) and prototyping
├── src/                # Core source code
│   ├── sampling.py     # Frame extraction and dataset balancing
│   ├── model.py        # CNN Architecture definitions
│   └── train.py        # Training and validation loops
├── app.py              # Real-time webcam inference script
├── requirements.txt    # Production dependencies
└── README.md           # Project documentation