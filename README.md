# ACL Injury Detection from Knee MRI

This project focuses on detecting Anterior Cruciate Ligament (ACL) injuries from knee MRI scans using deep learning models.

## Overview

 Developed CNN and hybrid CNN-ViT models using PyTorch
 Achieved AUC of 0.897 through iterative model optimization
 Applied attention mechanisms and Grad-CAM for model interpretability

## Tech Stack

 Python
 PyTorch
 NumPy, Pandas
 Matplotlib

## Dataset

This project uses the MRNet dataset (Stanford Knee MRI Dataset).
The dataset is not included in this repository.

## Results

### ROC Curve

![ROC Curve](assetsroc_curves.png)

### Confusion Matrix

![Confusion Matrix](assetsconfusion_matrix.png)

### Training Curve

![Training Curve](assetstraining_curves.png)

### Sample Visualization

![Sample](assetssample_visualization.png)

## How to Run

1. Install dependencies

```
pip install -r requirements.txt
```

2. Run training

```
python srctrain.py
```

## Project Structure

```
.
├── src
├── assets
├── datasetv1.py
├── explore_dataset.py
├── example.py
├── requirements.txt
└── README.md
```

## Author

Michael Wijaya
