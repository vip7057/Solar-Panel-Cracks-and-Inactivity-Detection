# Crack and Inactive Region Detection Using ResNet

This project involves detecting cracks and inactive regions in images using a deep learning model based on the ResNet architecture. The model is trained on grayscale images, which are converted to RGB and normalized before being passed through a modified ResNet-50 backbone. The project includes custom data loading, preprocessing, model architecture modifications, and training utilities.

## Features

- **Custom Dataset Loader**: Converts grayscale images to RGB and applies transformations such as horizontal/vertical flipping, color jittering, and normalization.
- **ResNet-50 Backbone**: Pretrained ResNet-50 model with modified final layers to adapt to binary classification (crack, inactive).
- **Weighted Loss Function**: Implements weighted BCEWithLogitsLoss to handle class imbalance in the dataset.
- **Early Stopping and Learning Rate Scheduler**: Uses ReduceLROnPlateau and early stopping to prevent overfitting.
- **Model Checkpointing**: Saves and restores the best model based on validation F1 score.
- **ONNX Export**: Supports exporting the best model as an ONNX file for deployment.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Exporting Model](#exporting-model)
- [References](#references)

## Installation

To set up the environment for this project, install the required dependencies by running:

```bash
pip install -r requirements.txt
---
## Requirements
- Python 3.7+
- PyTorch
- torchvision
- scikit-image
- pandas
- matplotlib
- scikit-learn
- tqdm
