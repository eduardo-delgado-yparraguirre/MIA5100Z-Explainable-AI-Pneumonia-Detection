# 🩺 Explainable AI for Pneumonia Detection from Chest X-Rays

<p align="center">

**Deep Learning • Medical Imaging • Explainable AI (XAI)**

*An educational project demonstrating how to interpret Convolutional Neural Networks using Grad-CAM, SHAP, and LIME.*

</p>

---

## 📖 Overview

Artificial Intelligence has demonstrated remarkable performance in medical image classification. However, most deep learning models operate as **black boxes**, making it difficult to understand **why** a prediction was made.

This project demonstrates how **Explainable Artificial Intelligence (XAI)** techniques can improve the transparency of a Convolutional Neural Network (CNN) trained to detect **Pneumonia** from **Chest X-Ray images**.

A **MobileNetV2** model is trained using **Transfer Learning**, and three popular explainability techniques are applied to the same trained model:

- 🔥 **Grad-CAM**
- 🟥 **SHAP**
- 🟩 **LIME**

The objective is to compare these techniques and better understand how deep learning models make medical imaging decisions.

---

# ✨ Features

| Feature | Status |
|:------------------------------|:------:|
| MobileNetV2 Transfer Learning | ✅ |
| CNN Binary Classification | ✅ |
| Model Evaluation | ✅ |
| Confusion Matrix | ✅ |
| Classification Report | ✅ |
| Grad-CAM Visualization | ✅ |
| SHAP Visualization | ✅ |
| Smoothed SHAP Visualization | ✅ |
| LIME Visualization | ✅ |
| Fully Commented Educational Code | ✅ |
| Ready for Teaching & Learning | ✅ |

---

# 🏗️ Project Workflow

```text
                   Chest X-Ray Dataset
                           │
                           ▼
                 MobileNetV2 CNN Training
                           │
                           ▼
                  Pneumonia Prediction
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
      Grad-CAM           SHAP             LIME
          │                │                │
          ▼                ▼                ▼
  Attention Maps     Pixel Attribution  Superpixel Importance
```

---

# 📂 Repository Structure

```text
MIA5100Z-Explainable-AI-Pneumonia-Detection

│
├── train_model_teaching.py
│
├── generate_gradcam_examples.py
├── generate_shap_examples.py
├── generate_shap_examples_smoothed.py
├── generate_lime_examples.py
│
├── examples/
│   ├── training_accuracy.png
│   ├── gradcam_six_examples_report.png
│   ├── shap_six_examples_smoothed_report.png
│   └── lime_six_examples_report.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 📊 Training Performance

<p align="center">
<img src="examples/training_accuracy.png" width="700">
</p>

The CNN was trained using **Transfer Learning** with **MobileNetV2**. Training and validation accuracy are monitored throughout the learning process to evaluate convergence and detect potential overfitting.

---

# 🔥 Grad-CAM

<p align="center">
<img src="examples/gradcam_six_examples_report.png" width="900">
</p>

### What does Grad-CAM show?

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of the X-ray that most influenced the CNN's prediction.

**Advantages**

- Excellent localization
- Easy to interpret
- Clinician-friendly
- Ideal for CNNs

---

# 🟥 SHAP

<p align="center">
<img src="examples/shap_six_examples_smoothed_report.png" width="900">
</p>

### What does SHAP show?

SHAP (SHapley Additive exPlanations) estimates the contribution of each image pixel to the prediction.

- 🔴 Red increases the probability of Pneumonia.
- 🔵 Blue decreases the probability of Pneumonia.

This repository also includes a **smoothed SHAP visualization** that produces more coherent anatomical regions for easier interpretation.

---

# 🟩 LIME

<p align="center">
<img src="examples/lime_six_examples_report.png" width="900">
</p>

### What does LIME show?

LIME (Local Interpretable Model-Agnostic Explanations) divides the image into **superpixels** and determines which regions most strongly support the model's prediction.

Unlike Grad-CAM, LIME is **model agnostic** and can be applied to many different machine learning models.

---

# 📈 Explainability Comparison

| Method | Strengths | Limitations |
|----------|-----------|-------------|
| **Grad-CAM** | Excellent localization, intuitive visualizations | CNN-specific |
| **SHAP** | Strong theoretical foundation, pixel-level attribution | Computationally expensive |
| **Smoothed SHAP** | Improved visual interpretation | Visualization post-processing |
| **LIME** | Model-agnostic, interpretable superpixels | Sensitive to segmentation parameters |

---

# 📥 Dataset

The Chest X-Ray dataset is **not included** in this repository.

Please download it from Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

After extracting, your project should look like:

```text
project/

│
├── chest_xray/
│   ├── train/
│   ├── val/
│   └── test/
│
├── train_model_teaching.py
└── ...
```

## Dataset Attribution

This project uses the **Chest X-Ray Images (Pneumonia)** dataset originally published by:

> Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*. Mendeley Data, V2.

If you use this project in your own work, please cite the original dataset authors.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/eduardo-delgado-yparraguirre/MIA5100Z-Explainable-AI-Pneumonia-Detection.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## 1. Train the CNN

```bash
python train_model_teaching.py
```

This generates:

```
pneumonia_mobilenetv2_model.keras
```

---

## 2. Generate Grad-CAM explanations

```bash
python generate_gradcam_examples.py
```

---

## 3. Generate SHAP explanations

```bash
python generate_shap_examples.py
```

---

## 4. Generate Smoothed SHAP explanations

```bash
python generate_shap_examples_smoothed.py
```

---

## 5. Generate LIME explanations

```bash
python generate_lime_examples.py
```

---

# 📚 Technologies

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- SHAP
- LIME
- Scikit-Learn
- Scikit-Image
- SciPy

---

# 🎓 Educational Purpose

This repository was created for educational purposes to demonstrate:

- Deep Learning
- Transfer Learning
- Medical Image Classification
- Explainable Artificial Intelligence (XAI)
- Model Interpretability
- Reproducible AI Experiments

The code is intentionally written with detailed comments to help students understand every step of the process.

---

# 🚀 Future Improvements

- Interactive Streamlit application
- Upload custom Chest X-Rays
- Integrated Grad-CAM + SHAP + LIME dashboard
- Vision Transformer (ViT) implementation
- Multi-class chest disease classification
- Additional explainability methods (Integrated Gradients, Score-CAM, Guided Backpropagation)

---

# 👨‍💻 Author

**Eduardo Delgado**

Master of Engineering — Artificial Intelligence  
University of Ottawa

> **Note:** This project was developed with the assistance of generative AI tools. The code was reviewed, tested, and adapted by the author for educational purposes.

---

# 📄 License

This project is released under the **MIT License**.

---

⭐ If you found this repository useful, please consider giving it a star!
