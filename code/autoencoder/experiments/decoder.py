import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

from code.deeplearning.models import *
from code.autoencoder.experiments.prep_data import *
from code.deeplearning.transforms import *

from ..experiments.prep_data import *
import torch.nn.functional as F
from tqdm import tqdm
from ...deeplearning.transforms import *
from datetime import datetime



# Set up directory structures and GPU/CPU/MPS device
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%s')
print(os.getcwd())
while not os.getcwd().endswith('meningioma'): os.chdir('..')
DEVICE = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
SEED = 0
torch.manual_seed(SEED)  # Set the seed for CPU random number generators
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Set the seed for GPU random number generators



ds = UnlabeledScansDataset(
    root_dir='BraTS-MEN-Train',
    use_cache = True,
    size=96, 
    transforms=transforms.Compose([
        Normalize2(mean=[0], std=[1]),
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
    ]))

train_loader, val_loader = get_loaders(ds, 0.2, 4)


model = CalabreseModelEncodr(input_channels=2, layer_layout=[1, 1, 2, 2], original_shape = 96, use_batch=False).to(DEVICE)
full_model_state = torch.load('code/deeplearning/weights/withskip.pth')
model.load_state_dict(full_model_state)
criterion = nn.MSELoss()

avg_loss = 0
model.eval()
with torch.no_grad():
    for batch in tqdm(val_loader, desc='Batch', total=len(val_loader), position=1, leave=False):
        # Grab the batch data
        X_batch = stack_volumes(batch['mris']).to(DEVICE)

        outputs = model.forward_autoencoder(X_batch)
        loss = criterion(X_batch, outputs)
        avg_loss += loss.item()


avg_loss /= len(val_loader)
print(avg_loss)

# %%
