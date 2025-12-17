# Semantic Segmentation using U-Net on Cityscapes Dataset

## 📌 Project Overview

This project implements a **Semantic Segmentation pipeline** using a **U-Net architecture** trained on the **Cityscapes dataset**.
The goal is to perform **pixel-wise classification** of urban street scenes, accurately segmenting objects such as roads, buildings, vehicles, pedestrians, and other city elements.

The project follows a **clean, production-style workflow**, including data preprocessing, model training, evaluation using appropriate segmentation metrics, and qualitative visualization of predictions.

---

## 🗂️ Dataset

* **Dataset**: Cityscapes
* **Type**: Urban street scene semantic segmentation
* **Input**: RGB images
* **Output**: Pixel-wise class masks
* **Number of Classes**: 8 semantic classes

> ⚠️ Due to licensing restrictions, the dataset is **not included** in this repository.

Official dataset:
  [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)

---

## 🧠 Model Architecture

* **Model**: U-Net
* **Encoder–Decoder structure**
* **Skip connections** for spatial feature preservation
* **Output activation**: Softmax
* **Loss function**: Sparse Categorical Crossentropy

📌 The architecture is optimized for **pixel-level accuracy and boundary preservation**.

---

## 📊 Evaluation Metrics

> ⚠️ Accuracy alone is **not sufficient** for semantic segmentation due to background dominance.

The model is evaluated using:

* **Mean Intersection over Union (mIoU)**
* **Per-Class IoU**
* **Dice Coefficient**
* **Validation Loss**

These metrics provide a more reliable assessment of segmentation quality, especially for **small objects and object boundaries**.

---

## 📈 Results

* **Validation Accuracy**: ~87%
* **Stable convergence with decreasing validation loss**
* **Clear qualitative improvements in object boundaries**
* **Strong generalization across validation samples**

📌 Sample results are provided in the `results/` directory.

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Prepare dataset

Download Cityscapes and place it according to the expected directory structure inside `dataset.py`.

### 3️⃣ Train the model

```bash
python src/train.py
```

### 4️⃣ Evaluate the model

```bash
python src/evaluate.py
```

### 5️⃣ Run inference

```bash
python src/predict.py
```

---

## 🧪 Key Highlights

✔️ End-to-end segmentation pipeline
✔️ Clean and modular code structure
✔️ Proper evaluation metrics (IoU & Dice)
✔️ Visualization of predictions
✔️ Production-ready workflow

---

## 🚀 Future Improvements

* Add **data augmentation** for better generalization
* Experiment with **pretrained encoders (EfficientNet, ResNet)**
* Implement **Focal Loss** to handle class imbalance
* Optimize inference speed for deployment

---

## 👤 Author

**Abdulrahman Ahmed**
Machine Learning / Computer Vision Engineer

📌 Feel free to connect or review the project!
