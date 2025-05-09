# 🗺️ Landmark Recognition in Azerbaijan — Deep Learning Project

This project showcases a deep learning pipeline I developed for classifying famous landmarks in Azerbaijan using **transfer learning with ResNet-18**.  
It was part of my Deep Learning course (CSCI 4701), and I worked on it as a core contributor in a two-person team.

---

## 🚀 Overview

- 📚 **Course**: CSCI 4701 – Deep Learning  
- 🧠 **Model**: ResNet-18 (pretrained on ImageNet)  
- 🏛️ **Classes**:
  - Maiden Tower  
  - Deniz Mall  
  - Shirvanshahlar Palace  
  - Dədə Qorqud Monument  
  - Heydar Aliyev Center  

---

## 📁 Dataset

The dataset consists of 738 manually collected images (train/test split) and was published on [Hugging Face](https://huggingface.co/datasets/khaleed-mammad/azerbaijan-landmarks-dataset).  
We collected images from different times of day (day/night) and converted them from `.heic` to `.jpg`.

- 📦 **Train**: ~586 images  
- 📦 **Test**: ~152 images  
- 🖼️ Format: JPG  
- 🗂️ Structure: 5 folders per split (one per class)

---

## 🧰 My Contributions

This project was a great opportunity to work end-to-end on a deep learning application.  
Here’s what I personally worked on:

- 🖼️ **Data Preprocessing & Organization**: Converted image formats, organized into `train/` and `test/`, helped upload to Hugging Face.
- 🧱 **Code Modularization**: Wrote `data_loader.py`, `model.py`, and `utils.py` for clean and reusable architecture.
- 🧪 **Training Support**: Assisted with training baseline and augmented models, tuning, and testing.
- 📊 **Result Analysis**: Helped compare performance metrics, visualize results, and draw conclusions.

---

## 🧠 Model Details

- 🔍 **Backbone**: ResNet-18 (frozen)
- 🧩 **Custom Head**: Final FC layer adapted for 5 classes
- 🎯 **Loss**: CrossEntropyLoss  
- ⚙️ **Optimizer**: Adam (LR = 0.0005)  
- 🧴 **Regularization**:
  - Dropout: `p=0.3`
  - Weight Decay: `1e-5`
- 🌀 **Data Augmentation**:
  - RandomResizedCrop, HorizontalFlip, ColorJitter, Rotation

---

## 📊 Results

| Model               | Train Accuracy | Test Accuracy |
|--------------------|----------------|---------------|
| Baseline (no aug)  | 94.37%         | 98.68%        |
| Augmented          | 86.86%         | 92.76%        |

> **Takeaway**: While augmentation improved generalization, the dataset's quality already gave very strong performance without it.