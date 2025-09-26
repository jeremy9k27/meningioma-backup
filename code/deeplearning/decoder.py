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


print(os.getcwd())
while not os.getcwd().endswith('meningioma'): os.chdir('..')
DEVICE = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
SEED = 0
torch.manual_seed(SEED)  # Set the seed for CPU random number generators
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Set the seed for GPU random number generators



# Create dataset, and then dataloaders
ds = MeningiomaDataset(
    task_name='Chr22q',
    pulse_sequences=['t1_post', 'flair'],
    seg_rois=[22],
    transforms=transforms.Compose([
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
        Normalize2(mean=[0], std=[1])]))

        

# set replacement to false so every sample appears exactly once
data = create_dataloaders(ds, bs=4, train_prop = 1, independent_test_set=True, seed=SEED, replacement = False)


model = CalabreseModelUNetSkip(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch=False).to(DEVICE)
full_model_state = torch.load('code/deeplearning/weights/unet.pth')
model.load_state_dict(full_model_state)
criterion = nn.MSELoss()

avg_loss = 0

model.eval()
with torch.no_grad():
    for batch in tqdm(data['train'], desc='Batch', total=len(data['train']), position=1, leave=False):
            # Grab the batch data
            X_batch = stack_volumes(batch['mris']).to(DEVICE)

            outputs = model.forward_autoencoder(X_batch)
            loss = criterion(X_batch, outputs)
            avg_loss += loss.item()

avg_loss /= len(data['train'])
print(avg_loss)

# %%
