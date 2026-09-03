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
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json


# Set up directory structures and GPU/CPU/MPS device
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%s')
OUTPUT_DIR = f'results_new/deeplearning/debugging/methyl_not_pretrained/run_{timestamp}'
print(os.getcwd())
while not os.getcwd().endswith('meningioma'): os.chdir('..')
DEVICE = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
SEED = 0
torch.manual_seed(SEED)  # Set the seed for CPU random number generators
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Set the seed for GPU random number generators
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def evaluate_on_unseen(model, criterion, dataloader, encoder = None):

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
            loss += criterion(outputs.squeeze(1), y_batch.long()).item()
    
    # Calculate evaluation metrics and return
    loss /= len(dataloader)

    preds = pd.DataFrame({
        'SubjectID': sub_IDs.cpu().numpy(),
        'y': y_trues.cpu().numpy(),
        'y_pred': y_preds.cpu().squeeze().numpy()
    })
    return preds


def train(model, optimizer, criterion, data, fold, epochs, unfreeze = False):

    tensorboard_writer = SummaryWriter(log_dir=f'{OUTPUT_DIR}/tensorboard_logs/{fold}')
    
    early_stopper = EarlyStopper(patience=10)
    best_val_acc = 0
    to_unfreeze = False
    # Loop thru all epochs
    #aug_transform = DetRotation3D(degrees=24)
    for epoch in tqdm(range(epochs), desc='Epoch', total=epochs):

        if to_unfreeze:
            # Unfreeze encoder
            print("unfroze")
            for param in model.encoder.parameters():
                param.requires_grad = True        
                    
            '''
            optimizer = optim.AdamW([
                            {'params': model.encoder.parameters(), 'lr': 0.00001},
                            {'params': model.classifier.parameters(), 'lr': 0.0001}
                        ], weight_decay=0.001)
            '''
                        
            optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
            to_unfreeze = False
            early_stopper = EarlyStopper(patience=10)
        

        train_loss = 0.
        model.train()
        y_preds, y_trues = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE)

        # Loop thru all batches
        for batch in tqdm(data['train'], desc='Batch', total=len(data['train']), position=1, leave=False):
            # Grab the batch data
            X_batch = stack_volumes(batch['mris']).to(DEVICE)
            #X_batch_aug = aug_transform(X_batch, batch['sub_id'], epoch).to(DEVICE) 
            y_batch = batch['label'].to(DEVICE)
            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)

            # Keep track of predictions and true labels
            y_preds = torch.cat((y_preds, outputs.squeeze(1).detach()))
            y_trues = torch.cat((y_trues, y_batch.detach()))
            # Backward pass
            loss = criterion(outputs.squeeze(1), y_batch.long())
            loss.backward()
            # Take an optimization step
            optimizer.step()
            # Keep track of training loss
            train_loss += loss.item()
        
        # Training metrics
        train_loss /= len(data['train'])
        y_preds = torch.argmax(y_preds, dim=1)
        train_metrics = {
            'loss': train_loss,
            'balancedacc': balanced_accuracy(y_trues, y_preds).item(),
        }

        # Log metrics
        tensorboard_writer.add_scalars('Train', train_metrics, epoch)


        # Validation phase
        model.eval()
        val_loss = 0.
        val_y_preds, val_y_trues, sub_IDs = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE), torch.tensor([])
        
        with torch.no_grad():
            for batch in data['val']:
                # Grab the batch data (no augmentation for validation)
                X_batch = stack_volumes(batch['mris']).to(DEVICE)
                y_batch = batch['label'].to(DEVICE)
                
                # Forward pass
                outputs = model(X_batch)
                
                # Keep track of predictions and true labels
                val_y_preds = torch.cat((val_y_preds, outputs.squeeze(1)))
                val_y_trues = torch.cat((val_y_trues, y_batch))
                sub_IDs = torch.cat((sub_IDs, batch['sub_id']))
                
                # Calculate loss
                val_loss += criterion(outputs.squeeze(1), y_batch.long()).item()
        
        # Validation metrics
        val_loss /= len(data['val'])
        val_y_preds = torch.argmax(val_y_preds, dim=1)
        val_metrics = {
            'loss': val_loss,
            'balancedacc': balanced_accuracy(val_y_trues, val_y_preds).item()
        }

        # Log validation metrics
        tensorboard_writer.add_scalars('Validation', val_metrics, epoch)
        
        if val_metrics['balancedacc'] >= best_val_acc:
            best_val_metrics = val_metrics
            best_val_acc = val_metrics['balancedacc']

        if early_stopper.should_stop(train_loss):
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
   
    #save
    model_state = model.state_dict()
    torch.save(model_state, f"code/deeplearning/weights/methyl_unfreeze_fold{fold}")
    return best_val_metrics


from sklearn.model_selection import StratifiedKFold

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
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(subject_ids, all_labels)):
        print(f"\n--- K-Fold {fold+1}/{n_splits} ---")
        
        # Split train and val subject IDs
        train_ids = [subject_ids[i] for i in train_idx]
        val_ids = [subject_ids[i] for i in val_idx]
        
        print(f"Train subjects: {len(train_ids)}, Val subjects: {len(val_ids)}")

        # Create dataloaders from manual splits
        dataloaders = create_only_train_val_dataloaders_loocv(
            ds,
            bs=4,
            train_ids=train_ids,
            val_ids=val_ids
        )

        model = CalabreseModelEncoder(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch=False, output_features=3, activation=None).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params}")

        
        '''
        full_model = CalabreseModelUNetSkip(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, pyrad_targets=18, use_batch=False).to(DEVICE)
        full_model.load_state_dict(torch.load('code/deeplearning/weights/unet.pth'))
        model.encoder = full_model.encoder
        del full_model
        torch.cuda.empty_cache()
        
        
        model.encoder.eval()
        for param in model.encoder.parameters():
            param.requires_grad = False  
        '''
        
        
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)

        criterion = nn.CrossEntropyLoss()

        # Train on fold training set and return best val predictions
        #preds_df = train(model, optimizer, criterion, dataloaders, fold, epochs = 100)
        best_metrics = train(model, optimizer, criterion, dataloaders, fold, unfreeze=True, epochs = 200)
        all_fold_metrics.append(best_metrics)

        # Evaluate on unseen data
        #preds_df = evaluate(model, criterion, dataloaders['val'])

        del model
        del optimizer
        torch.cuda.empty_cache()
    
    metrics_df = pd.DataFrame(all_fold_metrics)

    # Compute summary
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    summary_metrics = {}
    for metric in metrics_df.columns:
        summary_metrics[metric] = {
            "mean": mean_metrics[metric].item(),
            "std": std_metrics[metric].item()
        }

    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)

    print(f"Saved aggregated metrics to {metrics_path}")


ds = MeningiomaDataset(
    task_name='MethylationSubgroup',
    pulse_sequences=['t1_post', 'flair'],
    seg_rois=[22],
    transforms=transforms.Compose([
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
        Normalize2(mean=[0], std=[1])]))

ds.precache()

print("starting loocv")
kfold(ds)

# %%
