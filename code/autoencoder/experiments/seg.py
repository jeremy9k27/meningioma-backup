import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from ..models.models import *
from ..experiments.prep_data import *
import torch.nn.functional as F
from tqdm import tqdm
from ...deeplearning.transforms import *
from datetime import datetime

SEED = 0
torch.manual_seed(SEED)  # Set the seed for CPU random number generators
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Set the seed for GPU random number generators


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
OUTPUT_DIR = f'results_new/deeplearning/debugging/encoder-training/seg/run_{timestamp}'

# Hyperparameters
batch_size = 4
num_epochs = 500
output_path = f"code/deeplearning/weights/seg_dice.pth"
patience = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Dataset and Dataloaders
ds = UnlabeledScansDataset(
    root_dir= 'BraTS-MEN-Train', 
    use_pyrad=False,
    size=96, 
    transforms=transforms.Compose([
        Normalize2(mean=[0], std=[1]),
        CenterOnTumor(cube_size=96, margin=5, pad_size=60)]))

train_loader, val_loader = get_loaders(ds, 0.2, batch_size)

# Model, Loss, Optimizer
model = CalabreseModelUNetSkip(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch = False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)

# TensorBoard Writer
writer = SummaryWriter(log_dir=f'{OUTPUT_DIR}/tensorboard_logs')

# Early Stopper
early_stopper = EarlyStopper(patience=patience, min_delta=0.001)

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
        y_batch = batch['segs'][22].to(device)
        outputs = model.forward_seg(X_batch)

        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    base_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='val batches  '):

            X_batch = stack_volumes(batch['mris']).to(device)
            y_batch = batch['segs'][22].to(device)
            outputs = model.forward_seg(X_batch)

            baseline_value = X_batch.mean()
            baseline_pred = torch.full_like(X_batch, baseline_value)
            baseline_loss = criterion(baseline_pred, X_batch)

            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            base_loss += baseline_loss.item()


    val_loss /= len(val_loader)
    base_loss /= len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Base Loss = {base_loss:.4f}")

    # Log to TensorBoard
    writer.add_scalars('Loss', {
        'Train': train_loss,
        'Validation': val_loss
    }, epoch)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()

    if epoch % 10 == 0:
        print(f"saved at checkpoint {epoch}")
        torch.save(best_model_state, output_path)

    # Check early stopping
    if early_stopper.should_stop(val_loss):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Save the best model after early stopping
torch.save(best_model_state, output_path)
print(f"Best model saved to {output_path}")

# Close TensorBoard writer
writer.flush()
writer.close()