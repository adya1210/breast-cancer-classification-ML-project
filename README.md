# Breast Cancer Classification Across Magnification Levels

## Project Overview

This project investigates optimal combinations of image magnification and classification methods for automated breast cancer diagnosis. We compare five machine learning approaches across four magnification levels using the BreakHis histopathology dataset.

**Author:** Adya Verma  
**Course:** CS 6140 Machine Learning, Northeastern University  
**Date:** December 2025

## Key Findings

- **Best Model:** RBF SVM with transfer learning (F1: 97.98%)
- **Optimal Magnification:** 100X (best performance/cost balance)
- **Perfect Recall:** 100% at 40X and 100X (zero missed cancers)
- **Transfer Learning vs Deep Learning:** Transfer learning outperformed CNN trained from scratch by 5-10%

## Dataset

**BreakHis (Breast Cancer Histopathological Database)**
- 9,109 microscopy images from 82 patients
- 2,480 benign + 5,429 malignant samples
- 4 magnification levels: 40X, 100X, 200X, 400X
- 700×460 pixel RGB images, H&E stained

Download from: [Kaggle - BreakHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)

## Models Evaluated

### Traditional Machine Learning (with ResNet50 transfer learning)
1. **Logistic Regression** - L2 regularization, class weighting
2. **Linear SVM** - Maximum margin classifier
3. **RBF SVM** - Non-linear kernel, class weighting
4. **Random Forest** - Ensemble of decision trees

### Deep Learning
5. **Custom CNN** - 3 conv blocks, batch normalization, dropout

## Requirements
```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.5.0
tqdm>=4.65.0
```

## Installation
```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn Pillow tqdm
```

## Usage

### Running in Google Colab

1. Upload `breast_cancer_classification.ipynb` to Google Colab
2. Download BreakHis dataset from Kaggle
3. Upload dataset to Google Drive at: `/content/drive/MyDrive/archive (2)/BreaKHis_v1/`
4. Update `base_path` in notebook to match your Drive path
5. Runtime → Change runtime type → T4 GPU
6. Run all cells sequentially

## Results Summary

### Performance by Magnification (RBF SVM)

| Magnification | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| 40X | 96.33% | 94.93% | 100.00% | 97.40% |
| **100X** | **97.14%** | **96.05%** | **100.00%** | **97.98%** |
| 200X | 97.02% | 96.74% | 99.04% | 97.87% |
| 400X | 96.69% | 97.61% | 97.61% | 97.61% |
