###**Crack and Inactive Region Detection Using ResNet**
This project involves detecting cracks and inactive regions in images using a deep learning model based on the ResNet architecture. The model is trained on grayscale images, which are converted to RGB and normalized before being passed through a modified ResNet-50 backbone. The project includes custom data loading, preprocessing, model architecture modifications, and training utilities.

##**Features**
- Custom Dataset Loader: Uses grayscale images, converts them to RGB, and applies a series of transformations such as horizontal and vertical flipping, color jittering, and normalization.
- ResNet-50 Backbone: Pretrained ResNet-50 is used, with modifications to the final layers to adapt it to our binary classification problem (crack, inactive).
- Weighted Loss Function: Implements weighted loss (BCEWithLogitsLoss) to handle class imbalance in the dataset.
- Early Stopping and Learning Rate Scheduler: Automatically adjusts learning rate using ReduceLROnPlateau and implements early stopping to prevent overfitting.
- Model Checkpointing: Saves and restores the best model during training based on validation F1 score.
- ONNX Export: Supports saving the best model as an ONNX file for deployment.

