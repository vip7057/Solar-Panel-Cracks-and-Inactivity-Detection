import torch as t
from torch.nn import BCEWithLogitsLoss

from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Step 1: Load the data and perform train-test-split
data = pd.read_csv("data.csv", sep=';')
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

print("Training data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)

# Step 2: Set up data loading for the training and validation sets
train_dataset = ChallengeDataset(train_data, mode='train')
val_dataset = ChallengeDataset(val_data, mode='val')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 3: Create an instance of our ResNet model
resnet_model = model.ResNet()

# # Step 4: Set up loss criterion and optimizer
# criterion = t.nn.BCELoss()
# # criterion = t.nn.BCEWithLogitsLoss(weight=sample_weights.values)
# optimizer = t.optim.Adam(resnet_model.parameters(), lr=0.0001)# t.optim.SGD(resnet_model.parameters(), lr=0.0001, momentum=0.9)
# #lr = 0.00001 gave overfitting
# Step 4: Set up loss criterion and optimizer
class_counts = train_data[['crack', 'inactive']].sum(axis=0)
class_weights = 1.0 / class_counts.astype(float)
sample_weights = train_data[['crack', 'inactive']].idxmax(axis=1).map(class_weights)

# Convert NumPy array to PyTorch tensor
weight_tensor = t.tensor(sample_weights.values, dtype=t.float32)

criterion = BCEWithLogitsLoss(weight=weight_tensor)
optimizer = t.optim.Adam(resnet_model.parameters(), lr=0.0001)

# Step 5: Create a Trainer object and set its early stopping criterion
trainer = Trainer(model=resnet_model, crit=criterion, optim=optimizer, train_dl=train_loader, val_test_dl=val_loader,
                  cuda=True, early_stopping_patience=10)

# Step 6: Call the fit method on the trainer
train_losses, val_losses = trainer.fit(epochs=50)

# Access the val_f1_scores list from the trainer object
val_f1_scores = trainer.val_f1_scores

# Step 7: Find the epoch with the highest F1 score
best_epoch = np.argmax(val_f1_scores)
print(f"Best epoch: {best_epoch + 1}, F1 Score: {val_f1_scores[best_epoch]}")

# Step 8: Restore the best performing model checkpoint
trainer.restore_checkpoint(best_epoch)

# Step 9: Save the best performing model as ONNX with epoch number appended to filename
best_model_filename = f"best_model_epoch{best_epoch}.onnx"
trainer.save_onnx(best_model_filename)

# Step 10: Plot the results
plt.plot(np.arange(len(train_losses)), train_losses, label='train loss')
plt.plot(np.arange(len(val_losses)), val_losses, label='val loss')
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig('losses.png')
