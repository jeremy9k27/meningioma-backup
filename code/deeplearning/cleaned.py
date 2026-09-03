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
from deeplearning.utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json
import copy
import random
from sklearn.model_selection import StratifiedKFold

model_type = 'pretrained'
unfreeze = False
save_weights = True
aug = False
weights_dest = '22q/22q_unfreeze_constant_fold'
task = '22q'
scheduler_type = 'plateau'
lr = 0.0001
csv_file = 'results_new/deeplearning/debugging/lrs_val_loss.csv'
early_stopper_patience = 40
num_epochs = 200
bs = 4
SEED = 1

# Set up directory structures and GPU/CPU/MPS device
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%s')
OUTPUT_DIR = f'results_new/deeplearning/debugging/{model_type}/run_{timestamp}'
while not os.getcwd().endswith('meningioma'): os.chdir('..')
DEVICE = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')


def set_seeds():
    torch.manual_seed(SEED)  # Set the seed for CPU random number generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)  # Set the seed for GPU random number generators
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

def train_phase():
    train_loss = 0.
    model.train()
    y_preds, y_trues = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE)

    # Loop thru all batches
    for batch in tqdm(data['train'], desc='Batch', total=len(data['train']), position=1, leave=False):

        if aug: batch = aug_transform(batch)
        batch = BatchedCenterObj(batch)
        X_batch = stack_volumes(batch['mris']).to(DEVICE)


        y_batch = batch['label'].to(DEVICE)
        optimizer.zero_grad()

        outputs = model(X_batch)

        # Keep track of predictions and true labels
        y_preds = torch.cat((y_preds, outputs.squeeze(1).detach()))
        y_trues = torch.cat((y_trues, y_batch.detach()))
        loss = criterion(outputs.squeeze(1), y_batch.float())
        loss.backward()
        optimizer.step()
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
        'fdr': false_discovery_rate(y_trues, y_preds).item(),
        'lr': optimizer.param_groups[0]['lr'] 
    }

    # Log metrics
    tensorboard_writer.add_scalars('Train', train_metrics, epoch)
    
def val_phase():
    # Validation phase
    model.eval()
    val_loss = 0.
    val_y_preds, val_y_trues = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE)
    
    with torch.no_grad():
        for batch in data['val']:
            # Grab the batch data (no augmentation for validation)
            batch = BatchedCenterObj(batch)
            X_batch = stack_volumes(batch['mris']).to(DEVICE)
            y_batch = batch['label'].to(DEVICE)
            
            # Forward pass
            outputs = model(X_batch)           
            
            # Keep track of predictions and true labels
            val_y_preds = torch.cat((val_y_preds, outputs.squeeze(1)))
            val_y_trues = torch.cat((val_y_trues, y_batch))
            
            # Calculate loss
            val_loss += criterion(outputs.squeeze(1), y_batch.float()).item()
    
    # Validation metrics
    val_loss /= len(data['val'])
    val_metrics = {
        'loss': val_loss,
        'balancedacc': balanced_accuracy(val_y_trues, val_y_preds).item(),
        'aucpr': average_precision_score(val_y_trues.cpu().numpy(), val_y_preds.cpu().detach().numpy()),
        'auroc': roc_auc_score(val_y_trues.cpu().numpy(), val_y_preds.cpu().detach().numpy()),
        'tpr': true_positive_rate(val_y_trues, val_y_preds).item(),
        'fpr': false_positive_rate(val_y_trues, val_y_preds).item(),
        'fdr': false_discovery_rate(val_y_trues, val_y_preds).item()
    }

    # Log validation metrics
    tensorboard_writer.add_scalars('Validation', val_metrics, epoch)
    
    # Track best val loss
    if val_metrics['loss'] < best_loss:
        if save_weights: torch.save(model.state_dict(), f"code/deeplearning/weights/{weights_dest}{fold}")
        best_metrics = val_metrics
        best_loss = val_metrics['loss']
        
    return best_metrics
    

def train(model, optimizer, early_stopper, scheduler, criterion, data, fold, epochs, unfreeze = False):

    tensorboard_writer = SummaryWriter(log_dir=f'{OUTPUT_DIR}/tensorboard_logs/{fold}')
    to_unfreeze = False
    aug_transform = CalabreseRotation3D()
    CenterObj = CenterOnTumor(cube_size=96, margin=5, pad_size=60)
    BatchedCenterObj = BatchedCenterOnTumor(CenterObj)
    
    for epoch in tqdm(range(epochs), desc='Epoch', total=epochs):

        if to_unfreeze:
            print("unfroze")
            best_val_loss = 99 #forces us to save a model AFTER unfreezing
            for param in model.encoder.parameters():
                param.requires_grad = True        
            
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
            
            if scheduler_type == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

            elif scheduler_type == "plateau":
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=9, min_lr=1e-7) 
                         
            to_unfreeze = False
            early_stopper = EarlyStopper(patience=early_stopper_patience*2)
        
        
        train_phase()
        
        best_metrics = val_phase()


        if scheduler is not None and scheduler_type == "plateau": 
            scheduler.step(val_loss) 
        if scheduler is not None and scheduler_type == "cosine":
            scheduler.step()
        if early_stopper.should_stop(val_loss):
            if unfreeze:
                print(f"Encoder to unfreeze at epoch {epoch+2}")
                unfreeze = False
                to_unfreeze = True
            else:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
    # Close logging
    tensorboard_writer.flush()
    tensorboard_writer.close()
    
    return best_metrics


def kfold(ds: MeningiomaDataset, n_splits=5):
            
    all_preds = []
    all_trues = []
    all_ids = []

    all_fold_metrics = []

    # Ensure dataset is fully loaded
    ds.precache()
    
    # Get all subject IDs and labels for stratification
    subject_ids = ds.get_subjects()
    print(subject_ids)
    all_labels = [ds.get_labels().iloc[ds.get_subjects().index(sid)] for sid in subject_ids]
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(subject_ids, all_labels)):

        set_seeds()

        print(f"\n--- K-Fold {fold+1}/{n_splits} ---")
        
        # Split train and val subject IDs
        train_ids = [subject_ids[i] for i in train_idx]
        val_ids = [subject_ids[i] for i in val_idx]

        print(val_ids)
        print(val_idx)
        
        print(f"Train subjects: {len(train_ids)}, Val subjects: {len(val_ids)}")

        # Create dataloaders from manual splits
        dataloaders = create_only_train_val_dataloaders_loocv(
            ds,
            bs=bs,
            train_ids=train_ids,
            val_ids=val_ids
        )

        model = CalabreseModelEncoder(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch=False).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params}")

        
        #Initialize new model 
        full_model = CalabreseModelUNetSkip(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, pyrad_targets=18, use_batch=False).to(DEVICE)
        
        if model_type == 'pretrained':
            full_model.load_state_dict(torch.load('code/deeplearning/weights/unet.pth'))
            model.encoder = full_model.encoder
            del full_model
            torch.cuda.empty_cache() 
            
            model.encoder.eval()
            for param in model.encoder.parameters():
                param.requires_grad = False  
            
             
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
        early_stopper = EarlyStopper(patience=early_stopper_patience)
        if unfreeze == True:
            scheduler = None
            optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001) #just use constant initial if unfreezing

        elif scheduler_type == "constant": 
            scheduler = None
        
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        
        elif scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min',        
                factor=0.5,           # halve LR
                patience=4,
                min_lr=1e-7,          # don't go below this
            )  
        else:
            raise Exception("unkown lr scheduler")
        

        criterion = nn.BCELoss()

        # Train on fold training set and return best val predictions
        best_metrics = train(model, optimizer, early_stopper, scheduler, criterion, dataloaders, fold, unfreeze=unfreeze, epochs = num_epochs)
        all_fold_metrics.append(best_metrics)

        del model
        del optimizer
        torch.cuda.empty_cache()


    metrics_df = pd.DataFrame(all_fold_metrics)

    # Compute summary statistics
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    summary_metrics = {
        metric: {
            "mean": mean_metrics[metric].item(),
            "std": std_metrics[metric].item()
        }
        for metric in metrics_df.columns
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)

    print(f"Saved aggregated metrics to {metrics_path}")

    # Prepare row for CSV (means only)
    mean_metrics_dict = {
        metric: round(values["mean"], 3)
        for metric, values in summary_metrics.items()
    }

    metadata = {
        "task": task,
        "model_type": model_type,
        "scheduler": scheduler_type,
        "init_lr": lr,
        "aug": aug,
        "seed": SEED
    }

    full_row = {**metadata, **mean_metrics_dict}
    summary_row = pd.DataFrame([full_row])
    summary_row.to_csv(csv_file, mode="a", index=False, header=False)



ds = MeningiomaDataset(
    task_name='Chr22q',
    pulse_sequences=['t1_post', 'flair'],
    seg_rois=[22],
    transforms=transforms.Compose([
        Normalize2(mean=[0], std=[1]),
        ]))

ds.precache()

print("starting loocv")
kfold(ds)

# %%
