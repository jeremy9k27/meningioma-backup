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
OUTPUT_DIR = f'results_new/deeplearning/debugging/encoder-training/run_{timestamp}'

# Hyperparameters
batch_size = 4
num_epochs = 1000
output_path = f"code/deeplearning/weights/test_pyrad_1layer.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and Dataloaders
ds = UnlabeledScansDataset(
    root_dir='BraTS-MEN-Train', 
    use_pyrad= True,
    size=96, 
    transforms=transforms.Compose([
        Normalize2(mean=[0], std=[1]),
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
    ]))

train_loader, val_loader = get_loaders(ds, 0.2, bs=batch_size)

# Model, Loss, Optimizer
model = CalabreseModelEncoder(input_channels=2, layer_layout=[1,1,2,2], original_shape = 96, pyrad_targets=18, use_batch=False).to(device)

criterion = nn.MSELoss()
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.05)

# TensorBoard Writer
writer = SummaryWriter(log_dir=f'{OUTPUT_DIR}/tensorboard_logs')

# Early Stopper
early_stopper = EarlyStopper(patience=50000, min_delta=0.001)

# Track best model
best_val_loss = float('inf')
best_model_state = None


# Training Loop
for epoch in range(num_epochs):
    #print(f"Epoch: {epoch+1}")
    model.train()
    train_loss = 0.0
    baseline_loss_total = 0.0  

    total_samples = 0
    for batch in tqdm(train_loader, desc='train batches'):
        X_batch = stack_volumes(batch['mris']).to(device)
        y_batch = concat_pyrad_features(batch['pyrads']).to(device)
        outputs = model.forward_pyrad_1layer(X_batch)
        #outputs = model.forward(X_batch)
        loss = criterion(outputs, y_batch)

        baseline_preds = torch.zeros_like(y_batch)  # mean of standardized target = 0
        base_loss = criterion(baseline_preds, y_batch)
        
        optimizer.zero_grad()
        loss.backward()

        if False:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad_norm = {param.grad.norm().item():.6f}")

        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)  # multiply loss by batch size
        baseline_loss_total += base_loss.item() *  X_batch.size(0)  
        total_samples += X_batch.size(0)
    


    train_loss /= total_samples
    baseline_loss_total /= total_samples

    # Validation
    model.eval()
    val_loss = 0.0
    val_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='val batches  '):
            X_batch = stack_volumes(batch['mris']).to(device)
            y_batch = concat_pyrad_features(batch['pyrads']).to(device)

            outputs = model.forward_pyrad_1layer(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            val_samples += X_batch.size(0)

    val_loss /= val_samples

    print(f"Epoch {epoch+1}: Base = {baseline_loss_total:.4f}, Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Log to TensorBoard
    writer.add_scalars('Loss', {
        'Train': train_loss,
        'Validation': val_loss
    }, epoch)

    if epoch % 10 == 0:
        print(f"saved at checkpoint {epoch}")
        torch.save(model.state_dict(), output_path)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()

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