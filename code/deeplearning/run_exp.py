# %%
import os
while not os.getcwd().endswith('code'): os.chdir('..')
from preprocessing.utils import explore_3D_array_with_mask_contour
from deeplearning.transforms import CenterOnTumor, Normalize
from deeplearning.prep_data import MeningiomaDataset, create_dataloaders, create_only_train_val_dataloaders, stack_volumes
from deeplearning.models import CalabreseModel
from deeplearning.metrics import *
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

# Set up directory structures and GPU/CPU/MPS device
OUTPUT_DIR = 'results/deeplearning/debugging'
while not os.getcwd().endswith('Meningioma'): os.chdir('..')
DEVICE = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
SEED = 0
torch.manual_seed(SEED)  # Set the seed for CPU random number generators
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Set the seed for GPU random number generators

def evaluate(model, criterion, dataloader):
    # Setup for evaluation
    model.eval()
    with torch.no_grad():
        loss = 0.
        y_preds, y_trues, sub_IDs = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE), torch.tensor([])
        for batch in dataloader:
            # Grab the batch data
            X_batch = stack_volumes(batch['mris']).to(DEVICE)
            y_batch = batch['label'].to(DEVICE)
            # Run inference
            outputs = model(X_batch)
            # Keep track of predictions and true labels
            y_preds = torch.cat((y_preds, outputs.squeeze(1)))
            y_trues = torch.cat((y_trues, y_batch))
            sub_IDs = torch.cat((sub_IDs, batch['sub_id']))
            # Backward pass
            loss += criterion(outputs.squeeze(1), y_batch.float()).item()
    
    # Calculate evaluation metrics and return
    loss /= len(dataloader)
    metrics = {
        'loss': loss,
        'balancedacc': balanced_accuracy(y_trues, y_preds).item(),
        'aucpr': average_precision_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
        'auroc': roc_auc_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
        'tpr': true_positive_rate(y_trues, y_preds).item(),
        'fpr': false_positive_rate(y_trues, y_preds).item(),
        'fdr': false_discovery_rate(y_trues, y_preds).item()
    }
    preds = pd.DataFrame({
        'SubjectID': sub_IDs.cpu().numpy(),
        'y': y_trues.cpu().numpy(),
        'y_pred': y_preds.cpu().squeeze().numpy()
    })
    return metrics, preds

def train(model, optimizer, criterion, data, epochs=40):
    # Set up logging and metrics
    tensorboard_writer = SummaryWriter(log_dir=f'{OUTPUT_DIR}/tensorboard_logs')
    train_loss = 0.
    best_val_balanced_acc = 0.
    best_val_loss = float('inf')
    # Loop thru all epochs
    for epoch in tqdm(range(epochs), desc='Epoch', total=epochs):
        # Setup for the epoch
        model.train()
        y_preds, y_trues = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE)
        # Loop thru all batches
        for batch in tqdm(data['train'], desc='Batch', total=len(data['train']), position=1, leave=False):
            # Grab the batch data
            X_batch = stack_volumes(batch['mris']).to(DEVICE)
            y_batch = batch['label'].to(DEVICE)
            # Zero out the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(X_batch)
            # Keep track of predictions and true labels
            y_preds = torch.cat((y_preds, outputs.squeeze(1)))
            y_trues = torch.cat((y_trues, y_batch))
            # Backward pass
            loss = criterion(outputs.squeeze(1), y_batch.float())
            loss.backward()
            # Take an optimization step
            optimizer.step()
            # Keep track of training loss
            train_loss += loss.item()
        
        # Training metrics
        train_loss /= len(data['train'])
        train_metrics = {
            'loss': train_loss,
            'balancedacc': balanced_accuracy(y_trues, y_preds).item(),
            'aucpr': average_precision_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
            'auroc': roc_auc_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
            'tpr': true_positive_rate(y_trues, y_preds).item(),
            'fpr': false_positive_rate(y_trues, y_preds).item(),
            'fdr': false_discovery_rate(y_trues, y_preds).item()
        }

        # Validation metrics
        val_metrics, _ = evaluate(model, criterion, data['val'])

        # Log metrics
        tensorboard_writer.add_scalars('Train', train_metrics, epoch)
        tensorboard_writer.add_scalars('Val', val_metrics, epoch)

        # Save best performing models
        if not os.path.exists(f'{OUTPUT_DIR}/model_weights'): os.makedirs(f'{OUTPUT_DIR}/model_weights')
        if val_metrics['loss'] < best_val_loss:
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/model_weights/best_val_loss.pt')
            best_val_loss = val_metrics['loss']
        if val_metrics['balancedacc'] > best_val_balanced_acc:
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/model_weights/best_val_balancedacc.pt')
            best_val_loss = val_metrics['balancedacc']
        
    # Close logging
    tensorboard_writer.flush()
    tensorboard_writer.close()

# Create dataset, and then dataloaders
ds = MeningiomaDataset(
    task_name='Chr22q',
    pulse_sequences=['t1_post', 'flair', 'adc'],
    seg_rois=[22],
    transforms=transforms.Compose([
        Normalize(mean=[0], std=[1]),
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
    ])
)
ds.precache()
ds.plot_data_split()
dataloaders = create_dataloaders(ds, bs=4, independent_test_set=True, seed=SEED)

# Initialize model, optimizer, and loss fn
model = CalabreseModel(input_channels=3).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
criterion = nn.BCELoss()

# %%
# Train
train(model, optimizer, criterion, dataloaders)

# %%
# Test
eval_dict = {}
for weights in ['best_val_loss', 'best_val_balancedacc']:
    weights_path = f'{OUTPUT_DIR}/model_weights/{weights}.pt'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        eval_dict[weights] = {}
        for k, dataloader in dataloaders.items():
            eval_dict[weights][k], preds_df = evaluate(model, criterion, dataloader)
            preds_dir = f'{OUTPUT_DIR}/predictions/{weights}'
            if not os.path.exists(preds_dir): os.makedirs(preds_dir)
            preds_df.to_csv(f'{preds_dir}/{k}_preds.csv', index=False)

eval_dict
# %%
