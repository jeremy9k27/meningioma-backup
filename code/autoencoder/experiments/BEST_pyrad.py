# %%
# we will compute pyrad values here, and not used any cached results
while not os.getcwd().endswith('meningioma'): os.chdir('..')

print("start1", os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
import torch.nn.functional as F
from tqdm import tqdm
import sys

from code.deeplearning.BEST_models import *
from code.autoencoder.experiments.BEST_prep_data import *
from code.deeplearning.transforms import *
from datetime import datetime
print("start2", os.getcwd())

class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.001):
        """
        Early stopping checker
        
        Args:
            patience: Number of epochs to wait without improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def should_stop(self, val_loss):
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            # No improvement
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.patience
    
    def reset(self):
        """Reset the early stopper"""
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

# Set output directory
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
OUTPUT_DIR = f'results_new/deeplearning/debugging/encoder-training/BEST_run_{timestamp}'

# Hyperparameters
batch_size = 4
num_epochs = 200
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and Dataloaders
ds = UnlabeledScansDataset('BraTS-MEN-Train', 
    size=96, 
    transforms=transforms.Compose([
        Normalize2(mean=[0], std=[1]),
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
    ]),
    timestamp=timestamp)

train_loader, val_loader = get_loaders(ds, 0.1, batch_size)

# Model, Loss, Optimizer
model = CalabreseModelEncoder(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, pyrad_targets=24).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard Writer
writer = SummaryWriter(log_dir=f'{OUTPUT_DIR}/tensorboard_logs')

# Early Stopper
early_stopper = EarlyStopper(patience=4, min_delta=0.001)

# Track best model
best_val_loss = float('inf')
best_model_state = None

# Training Loop
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}")
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc='train batches'):
        X_batch = stack_volumes(batch['mris']).to(device)
        y_batch = concat_pyrad_features(batch['pyrads']).to(device)
        outputs = model.forward_pyrad_2layer(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='val batches  '):
            X_batch = stack_volumes(batch['mris']).to(device)
            y_batch = concat_pyrad_features(batch['pyrads']).to(device)
            outputs = model.forward_pyrad_1layer(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Log to TensorBoard
    writer.add_scalars('Loss', {
        'Train': train_loss,
        'Validation': val_loss
    }, epoch)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.encoder.state_dict()

    # Check early stopping
    if early_stopper.should_stop(val_loss):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Save the best model after early stopping
output_path = f"code/deeplearning/weights/BEST_pyrad_2layer.pth"
torch.save(best_model_state, output_path)
print(f"Best model saved to {output_path}")

# Close TensorBoard writer
writer.flush()
writer.close()
# %%
