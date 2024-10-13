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

### Uasing requirements.txt
```bash
pip install -r requirements.txt
```
### Using environment.yml
Alternatively, you can create the environment using Conda with the following command:

```bash
conda env create -f environment.yml
```

---
## Usage
**Data Preparation**
The dataset should be provided as a CSV file (data.csv) with the following columns:

- **Image Path:** Path to the image files.
- **Crack Label:** Binary label (1 for crack, 0 for no crack).
- **Inactive Label:** Binary label (1 for inactive, 0 for active).

Ensure the images are in grayscale format. The code automatically handles the conversion to RGB.


**Training**
To train the model, run:

```bash
python train.py
```
This will:

- Load the dataset from data.csv.
- Perform a 90-10 train-validation split.
- Train the model using the defined ResNet architecture.
- Apply early stopping based on the validation F1 score.
- Save the best model checkpoints.


---
## Project Structure
```plaintext
.
├── checkpoints/      # Folder to store model checkpoints
├── images/           # Folder containing training images
├── data.csv          # Folder containing image paths and corresponding labels
├── checkpoints      # Folder to store model checkpoints
├── data.py           # Data loader and preprocessing logic
├── model.py          # ResNet-50 based model architecture
├── train.py          # Main script to train the model
├── trainer.py        # Trainer class handling training and evaluation
└── requirements.txt  # Required dependencies
```
---
## Dataset
The dataset consists of grayscale images that are converted to RGB before being fed into the model. The dataset is expected to have two labels:

- **Crack:** Binary label indicating the presence of cracks.
- **Inactive:** Binary label indicating inactive regions.

**Data Augmentation**
- Horizontal and Vertical Flip: Applied with a probability of 0.4 to improve model robustness.
- Color Jittering: Random adjustments to brightness, contrast, saturation, and hue.
- Normalization: Images are normalized using the mean and standard deviation of the training set.

---
## Model Architecture
The model is based on the ResNet-50 architecture, pretrained on ImageNet. The final layers are modified for binary classification:

```python
self.fc1 = torch.nn.Linear(2048, 512)
self.fc2 = torch.nn.Linear(512, 256)
self.fc3 = torch.nn.Linear(256, 2)  # Output for two binary classification tasks
```
**Activation and Output**
The final layer uses a sigmoid activation to output probabilities for the two binary classification tasks.
---
## Training
The training process uses the following components:
- **Loss Function:** Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss) with class weights to handle dataset imbalance.
- **Optimizer:** Adam optimizer (torch.optim.Adam) with a learning rate of 0.0001.
- **Learning Rate Scheduler:** Reduces learning rate if the validation F1 score plateaus.
- **Batch Size:** Default is 32.
- **Early Stopping:** Training will stop if the validation F1 score does not improve for 10 consecutive epochs to prevent overfitting.

---
## Evaluation
The validation process computes:

**Loss:** Binary cross-entropy loss.
**F1 Score:** The F1 score is used as the primary metric for evaluating model performance.
The best model based on the validation F1 score is saved. The F1 score is computed as:

```python
f1_score(true_labels, predictions, average='weighted')
```
---
## Results
After training, the model's performance (loss, F1 score) is visualized with a loss plot saved as losses.png. The best model is saved in the checkpoints/ directory.
---
## Exporting Model
The best-performing model can be exported as an ONNX file for deployment:

```bash
python train.py
```
At the end of the training, the model is saved as an ONNX file:

```bash
best_model_epoch{epoch_number}.onnx
```
---
##References
ResNet Architecture: Deep Residual Learning for Image Recognition
PyTorch ONNX Export: PyTorch Documentation
