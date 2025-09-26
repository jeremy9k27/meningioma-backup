# %%
import os
if os.getcwd().endswith('meningioma'):
    os.chdir('code')
while not os.getcwd().endswith('code'): os.chdir('..')
import sys
sys.path.append(os.getcwd())
from preprocessing.utils import explore_3D_array_with_mask_contour
from deeplearning.transforms import *
from deeplearning.prep_data import *
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
OUTPUT_DIR = f'results_new/deeplearning/debugging/val-debug/run_{timestamp}'
print(os.getcwd())
while not os.getcwd().endswith('meningioma'): os.chdir('..')
DEVICE = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
SEED = 0
torch.manual_seed(SEED)  # Set the seed for CPU random number generators
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Set the seed for GPU random number generators
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            #pyrads = batch['pyrads'].to(DEVICE)
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
    aug_transform = DetRotation3D(degrees=30)

    best_val_balanced_acc = 0.
    best_val_loss = float('inf')
    # Loop thru all epochs

    for epoch in tqdm(range(epochs), desc='Epoch', total=epochs):
        # Setup for the epoch
        if epoch == -1:
            # Unfreeze encoder
            for param in model.encoder.parameters():
                param.requires_grad = True
            
            optimizer = optim.AdamW([
                            {'params': model.encoder.parameters(), 'lr': 0.00001},
                            {'params': model.classifier.parameters(), 'lr': 0.0001}
                        ], weight_decay=0.001)
        
        train_loss = 0.
        model.train()
        y_preds, y_trues = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE)
        # Loop thru all batches
        for batch in tqdm(data['train'], desc='Batch', total=len(data['train']), position=1, leave=False):
            # Grab the batch data
            X_batch = stack_volumes(batch['mris'])
            X_batch_aug = aug_transform(X_batch, batch['sub_id'], epoch).to(DEVICE) 
            y_batch = batch['label'].to(DEVICE)
            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch_aug)


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

        # Log training metrics
        tensorboard_writer.add_scalars('Train', train_metrics, epoch)
        
        # Mini-validation evaluation (only addition)
        if 'mini_val' in data:
            model.eval()
            val_loss = 0.
            val_y_preds, val_y_trues = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE)
            
            with torch.no_grad():
                for batch in data['mini_val']:
                    X_batch = stack_volumes(batch['mris']).to(DEVICE)
                    y_batch = batch['label'].to(DEVICE)
                    outputs = model(X_batch)
                    val_y_preds = torch.cat((val_y_preds, outputs.squeeze(1)))
                    val_y_trues = torch.cat((val_y_trues, y_batch))
                    val_loss += criterion(outputs.squeeze(1), y_batch.float()).item()
            
            val_loss /= len(data['mini_val'])
            val_metrics = {
                'loss': val_loss,
                'balancedacc': balanced_accuracy(val_y_trues, val_y_preds).item(),
                'aucpr': average_precision_score(val_y_trues.cpu().numpy(), val_y_preds.cpu().detach().numpy()),
                'auroc': roc_auc_score(val_y_trues.cpu().numpy(), val_y_preds.cpu().detach().numpy()),
                'tpr': true_positive_rate(val_y_trues, val_y_preds).item(),
                'fpr': false_positive_rate(val_y_trues, val_y_preds).item(),
                'fdr': false_discovery_rate(val_y_trues, val_y_preds).item()
            }
            tensorboard_writer.add_scalars('MiniVal', val_metrics, epoch)
        
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
        
        # Split available subjects
        available_train_ids = [s for s in subject_ids if s != val_id]
        val_ids = [val_id]
        
        # Create mini-val split
        import random
        random.shuffle(available_train_ids)
        split_idx = int(len(available_train_ids) * 0.85)  # 85% train, 15% mini-val
        actual_train_ids = available_train_ids[:split_idx]
        mini_val_ids = available_train_ids[split_idx:]
        
        print(f"  Train: {len(actual_train_ids)}, Mini-val: {len(mini_val_ids)}, Test: {len(val_ids)}")

        # Create dataloaders using your existing function structure
        dataloaders = create_train_minival_val_dataloaders_loocv(
            ds,
            bs=4,
            train_ids=actual_train_ids,
            mini_val_ids=mini_val_ids, 
            val_ids=val_ids
        )

        model = CalabreseModelEncoder(dense_features = 32 ,input_channels=3, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch=False).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params}")
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
        criterion = nn.BCELoss()

        # Train with mini-val monitoring (your modified train function)
        train(model, optimizer, criterion, dataloaders, i, epochs=50)

        # Evaluate on left-out subject  
        preds_df = evaluate(model, criterion, dataloaders['val'])

        # Rest of your existing accumulation code...
        all_preds.extend(preds_df['y_pred'].tolist())
        all_trues.extend(preds_df['y'].tolist()) 
        all_ids.extend(preds_df['SubjectID'].tolist())
        if i == -1:
            break

    # Rest of your existing code unchanged...
    # [All the metrics calculation, saving, etc. stays exactly the same]
    
    # Convert predictions and true labels to tensors
    y_pred_binary = (torch.tensor(all_preds) > 0.5).float()

    results_df = pd.DataFrame({
        'SubjectID': all_ids,
        'y_true': all_trues,
        'y_pred': all_preds,
        'y_pred_binary': y_pred_binary.numpy(),
        'correct': (torch.tensor(all_trues) == y_pred_binary).numpy(),
        'error_type': ['Correct' if (torch.tensor(all_trues)[i] == y_pred_binary[i]) 
                    else ('False Positive' if (torch.tensor(all_trues)[i] == 0 and y_pred_binary[i] == 1)
                    else 'False Negative') for i in range(len(all_trues))],
        'confidence': [abs(pred - 0.5) for pred in all_preds]
    })
    results_df = results_df.sort_values(['correct', 'confidence'], ascending=[True, False])

    y_preds = torch.tensor(all_preds, dtype=torch.float32, device=DEVICE)
    y_trues = torch.tensor(all_trues, dtype=torch.float32, device=DEVICE)

    bce_loss_fn = nn.BCELoss()
    avg_loss = bce_loss_fn(y_preds, y_trues).item()

    error_summary = results_df['error_type'].value_counts().to_dict()

    metrics = {
        'loss': avg_loss,
        'basicacc': basic_accuracy(y_trues, y_preds).item(),
        'balancedacc': balanced_accuracy(y_trues, y_preds).item(),
        'aucpr': average_precision_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
        'auroc': roc_auc_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
        'tpr': true_positive_rate(y_trues, y_preds).item(),
        'fpr': false_positive_rate(y_trues, y_preds).item(),
        'fdr': false_discovery_rate(y_trues, y_preds).item(),
        'error_breakdown': error_summary,
        'num_false_positives': error_summary.get('False Positive', 0),
        'num_false_negatives': error_summary.get('False Negative', 0),
        'total_errors': error_summary.get('False Positive', 0) + error_summary.get('False Negative', 0)
    }

    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    results_path = os.path.join(OUTPUT_DIR, 'detailed_results.csv')
    results_df.to_csv(results_path, index=True, index_label='original_index')

    print(f"\nError Analysis:")
    print(f"Total Correct: {error_summary.get('Correct', 0)}")
    print(f"False Positives: {metrics['num_false_positives']}")
    print(f"False Negatives: {metrics['num_false_negatives']}")
    print(f"Total Errors: {metrics['total_errors']}")

    print(f"\nWorst predictions (errors with lowest confidence):")
    error_cases = results_df[results_df['correct'] == False]
    if len(error_cases) > 0:
        print(error_cases[['SubjectID', 'y_true', 'y_pred', 'error_type', 'confidence']].head(10))
    else:
        print("No errors found!")

    return metrics, results_df
# Create dataset, and then dataloaders
ds = MeningiomaDataset(
    task_name='Chr22q',
    pulse_sequences=['t1_post', 'flair', 'adc'],
    seg_rois=[22],
    transforms=transforms.Compose([
        Normalize2(mean=[0], std=[1]),
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),])
)
ds.precache()

print("starting loocv")
loocv(ds)

# %%
