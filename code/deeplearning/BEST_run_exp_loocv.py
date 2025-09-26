# %%
import os
if os.getcwd().endswith('meningioma'):
    os.chdir('code')
while not os.getcwd().endswith('code'): os.chdir('..')
import sys
sys.path.append(os.getcwd())
from preprocessing.utils import explore_3D_array_with_mask_contour
from deeplearning.transforms import *
from deeplearning.prep_data import MeningiomaDataset, create_dataloaders, create_only_train_val_dataloaders, create_only_train_val_dataloaders_loocv, stack_volumes
from deeplearning.models import *
from deeplearning.metrics import *
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json


# Set up directory structures and GPU/CPU/MPS device
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%s')
OUTPUT_DIR = f'results_new/deeplearning/debugging/not_pretrained/run_{timestamp}'
print(os.getcwd())
while not os.getcwd().endswith('meningioma'): os.chdir('..')
DEVICE = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
SEED = 0
torch.manual_seed(SEED)  # Set the seed for CPU random number generators
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Set the seed for GPU random number generators

def evaluate(model, criterion, dataloader, encoder = None):

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

    preds = pd.DataFrame({
        'SubjectID': sub_IDs.cpu().numpy(),
        'y': y_trues.cpu().numpy(),
        'y_pred': y_preds.cpu().squeeze().numpy()
    })
    return preds

def train(model, optimizer, criterion, data, fold, epochs=40, encoder = None):

    # Set up logging and metrics
    tensorboard_writer = SummaryWriter(log_dir=f'{OUTPUT_DIR}/tensorboard_logs/{fold}')
    
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

        # Log metrics
        tensorboard_writer.add_scalars('Train', train_metrics, epoch)
        
        
    # Close logging
    tensorboard_writer.flush()
    tensorboard_writer.close()


def loocv(ds: MeningiomaDataset):
            
    all_preds = []
    all_trues = []
    all_ids = []

    # Ensure dataset is fully loaded
    ds.precache()
    
    # Get all subject IDs (assuming 1 sample per subject or known mapping)
    subject_ids = ds.get_subjects()

    for i, val_id in enumerate(subject_ids):
        print(f"\n--- LOOCV Fold {i+1}/{len(subject_ids)}; Val Subject: {val_id} ---")
        
        # Split train and val subject IDs
        train_ids = [s for s in subject_ids if s != val_id]
        val_ids = [val_id]

        # Create dataloaders from manual splits
        dataloaders = create_only_train_val_dataloaders_loocv(
            ds,
            bs=4,
            train_ids=train_ids,
            val_ids=val_ids
        )

        # Initialize new model + optimizer
        model = CalabreseModelEncoder(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96).to(DEVICE)
        model.encoder.load_state_dict(torch.load('code/deeplearning/weights/BEST-pyrad_2layer.pth'))
        
        model.encoder.eval()
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
        criterion = nn.BCELoss()

        # Train on N-1
        train(model, optimizer, criterion, dataloaders, i, epochs = 40)

        # Evaluate on left-out subject
        preds_df = evaluate(model, criterion, dataloaders['val'])

        # Accumulate predictions
        all_preds.extend(preds_df['y_pred'].tolist())
        all_trues.extend(preds_df['y'].tolist())
        all_ids.extend(preds_df['SubjectID'].tolist())
        if i == -1:
            break

    # After LOOCV
    results_df = pd.DataFrame({
        'SubjectID': all_ids,
        'y_true': all_trues,
        'y_pred': all_preds
    })

    # Convert predictions and true labels to tensors
    y_preds = torch.tensor(all_preds, dtype=torch.float32, device=DEVICE)
    y_trues = torch.tensor(all_trues, dtype=torch.float32, device=DEVICE)

    # Compute final average BCELoss
    bce_loss_fn = nn.BCELoss()
    avg_loss = bce_loss_fn(y_preds, y_trues).item()

    # Final evaluation metrics
    metrics = {
        'loss': avg_loss,
        'basicacc': basic_accuracy(y_trues, y_preds).item(),
        'balancedacc': balanced_accuracy(y_trues, y_preds).item(),
        'aucpr': average_precision_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
        'auroc': roc_auc_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
        'tpr': true_positive_rate(y_trues, y_preds).item(),
        'fpr': false_positive_rate(y_trues, y_preds).item(),
        'fdr': false_discovery_rate(y_trues, y_preds).item()
    }

    # Save metrics as JSON
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics, results_df


# Create dataset, and then dataloaders
ds = MeningiomaDataset(
    task_name='Chr22q',
    pulse_sequences=['t1_post', 'flair'],
    seg_rois=[22],
    transforms=transforms.Compose([
        Normalize2(mean=[0], std=[1]),
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
    ])
)
ds.precache()

print("starting loocv")
loocv(ds)

# %%
