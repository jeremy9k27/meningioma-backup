# %%
import os
if os.getcwd().endswith('meningioma'):
    os.chdir('code')
while not os.getcwd().endswith('code'): os.chdir('..')
import sys
sys.path.append(os.getcwd())
from preprocessing.utils import explore_3D_array_with_mask_contour
from deeplearning.transforms import *
from deeplearning.prep_data import MeningiomaDataset, create_dataloaders, create_only_train_val_dataloaders, stack_volumes
from deeplearning.models import CalabreseModelEncoder
from deeplearning.metrics import *
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

model = CalabreseModelEncoder(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch=False)

x = torch.zeros(4, 2, 96, 96, 96)
out = model.forward_autoencoder(x)

# %%
