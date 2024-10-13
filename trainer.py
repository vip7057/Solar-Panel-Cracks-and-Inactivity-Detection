import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau ###  LR scheduler
import numpy as np

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1  #The patience for early stopping
                 ):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        self._weight_decay = 1e-6   #L2 regularization 1: 1e-5 2: 1e-3 -> bad results for training 3: 1e-4


        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

        # Initialize learning rate scheduler
        self.lr_scheduler = ReduceLROnPlateau(self._optim, mode='max', factor=0.1, patience=1, verbose=True)

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})

    def train_step(self, x, y):
        self._optim.zero_grad()
        if self._cuda:
            x, y = x.cuda(), y.cuda()
        outputs = self._model(x)
###########
        # Ensure that the dimensions match
        if outputs.shape != y.shape:
            # If the dimensions don't match, you may need to reshape or adjust y accordingly
            y = y.view(outputs.shape)

        # Compute loss
        loss = self._crit(outputs, y)
            ############

        #loss = self._crit(t.squeeze(outputs).float(), y)

        # # Adding L2 regularization
        # l2_reg = t.tensor(0., device=x.device)
        # for param in self._model.parameters():
        #     l2_reg += t.norm(param)
        # loss += self._weight_decay * l2_reg

        loss.backward()
        self._optim.step()
        return loss.item()

    def val_test_step(self, x, y):
        with t.no_grad():
            if self._cuda:
                x, y = x.cuda(), y.cuda()
            outputs = self._model(x)
            loss = self._crit(t.squeeze(outputs).float(), y)
            predictions = outputs.cpu()
            #print(predictions)
            return loss.item(), predictions

    def train_epoch(self):
        self._model = self._model.train()
        total_loss = 0.0
        num_batches = len(self._train_dl)
        for batch in tqdm(self._train_dl, total=num_batches, desc="Training"):
            x, y = batch
            loss = self.train_step(x, y)
            total_loss += loss
        return total_loss / num_batches

    def val_test(self):
        self._model = self._model.eval()
        total_loss = 0.0
        predictions = []
        true_labels = []
        num_batches = len(self._val_test_dl)
        for batch in tqdm(self._val_test_dl, total=num_batches, desc="Validation/Test"):
            x, y = batch
            loss, outputs = self.val_test_step(x, y)  # Call val_test_step to get loss and predictions
            total_loss += loss
            predictions.extend(np.array(t.squeeze(outputs).round()))

            #print(predictions)
            true_labels.extend(np.array(y.cpu()))

        avg_loss = total_loss / num_batches
        f1 = f1_score(true_labels, predictions, average='weighted')
        self.f1 = f1
        print(f"Validation/Test Loss: {avg_loss}, F1 Score: {f1}")
        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        train_losses = []
        val_losses = []
        self.val_f1_scores = []
        high_f1_epochs = []
        epoch_counter = 0

        while True:
            if epochs != -1 and epoch_counter >= epochs:
                break

            # Train for one epoch
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            # Calculate validation loss and F1 score
            val_loss = self.val_test()
            val_f1 = self.f1
            val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)

            # Get current learning rate from optimizer
            current_lr = self._optim.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")
            # Update learning rate scheduler based on validation F1 score
            self.lr_scheduler.step(val_f1)


            # Save checkpoint if F1 score is above 0.7
            if val_f1 > 0.7:
                self.save_checkpoint(epoch_counter)
                print(f"Checkpoint saved. F1 score: {val_f1}")
                high_f1_epochs.append((epoch_counter, val_f1))

            # Early stopping check
            if self._early_stopping_patience != -1:
                if len(val_losses) > self._early_stopping_patience:
                    if all(val_losses[-1] >= loss for loss in val_losses[-(self._early_stopping_patience + 1):-1]):
                        print(
                            f"Stopping early. No improvement in validation loss for {self._early_stopping_patience} epochs.")
                        break

            # Print train loss, validation loss, and F1 score
            print(f"Epoch {epoch_counter + 1}/{epochs if epochs != -1 else '∞'} - Train Loss: {train_loss}, Validation Loss: {val_loss}, F1 Score: {val_f1}")

            epoch_counter += 1

            # Write epochs with F1 score > 0.8 to a text file
            with open("high_f1_epochs.txt", "w") as f:
                for epoch, f1_score in high_f1_epochs:
                    f.write(f"Epoch: {epoch}, F1 Score: {f1_score}\n")

        return train_losses, val_losses





