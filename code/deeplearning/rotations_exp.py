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
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from datetime import datetime
import json
import copy
import random
import logging
from pathlib import Path

bs = 4  # Can now be any batch size
SEED = 0 

# Set up directory structures and GPU/CPU/MPS device
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%s')
TIMESTAMP = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
print(os.getcwd())
while not os.getcwd().endswith('meningioma'): os.chdir('..')
DEVICE = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')

# Setup logging
os.makedirs("rotation_exp", exist_ok=True)
LOGFILE = f"rotation_exp/rotation_experiment_{TIMESTAMP}.log"
logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info(f"<>" * 40)
logging.info(f"Log file for {Path(__file__).name} run at {TIMESTAMP}")
logging.info(f"Batch size: {bs}")
logging.info(f"Seed: {SEED}")
logging.info(f"Device: {DEVICE}")
logging.info(f"Rotation angles: {(0,45,90)}")
logging.info(f"<>" * 40)

ds = MeningiomaDataset(
    task_name='Chr22q',
    pulse_sequences=['t1_post', 'flair'],
    seg_rois=[22],
    transforms=transforms.Compose([
        Normalize2(mean=[0], std=[1]),
        ]))

dataloaders = create_dataloaders(ds, bs=bs, independent_test_set=True, seed=SEED)

CalObj = CalabreseRotation3D()
CenterObj = CenterOnTumor(cube_size=96, margin=5, pad_size=60)
BatchedCenterObj = BatchedCenterOnTumor(CenterObj)

angles = (0,45,90)

step = 0
for batch in dataloaders['train']:
    print(step)
    no_rotate_batch = BatchedCenterObj(batch)
    
    batch_size = len(batch["sub_id"])
    
    for i in range(batch_size):
        pid = str(int(batch["sub_id"][i]))
        os.makedirs(f"rotation_exp/{pid}", exist_ok=True)
        os.makedirs(f"rotation_exp/{pid}/orig", exist_ok=True)
        os.makedirs(f"rotation_exp/{pid}/orig_no_center", exist_ok=True)
        os.makedirs(f"rotation_exp/{pid}/rotated", exist_ok=True)

        # save non-rotated, non-centered
        orig_nc_t1_post = batch['mris']['t1_post'][i].cpu().numpy()
        nifti_img = nib.Nifti1Image(orig_nc_t1_post, affine=np.eye(4))
        nib.save(nifti_img, f"rotation_exp/{pid}/orig_no_center/orig_nc_t1_post_{pid}.nii.gz")
        
        orig_nc_flair = batch['mris']['flair'][i].cpu().numpy()
        nifti_img = nib.Nifti1Image(orig_nc_flair, affine=np.eye(4))
        nib.save(nifti_img, f"rotation_exp/{pid}/orig_no_center/orig_nc_flair_{pid}.nii.gz")
        
        orig_nc_seg = batch['segs'][22][i].cpu().numpy().astype(np.int16)
        nifti_img = nib.Nifti1Image(orig_nc_seg, affine=np.eye(4))
        nib.save(nifti_img, f"rotation_exp/{pid}/orig_no_center/orig_nc_seg_{pid}.nii.gz")
        

        # save non-rotated, centered
        orig_t1_post = no_rotate_batch['mris']['t1_post'][i].cpu().numpy()
        nifti_img = nib.Nifti1Image(orig_t1_post, affine=np.eye(4))
        nib.save(nifti_img, f"rotation_exp/{pid}/orig/orig_t1_post_{pid}.nii.gz")
        
        orig_flair = no_rotate_batch['mris']['flair'][i].cpu().numpy()
        nifti_img = nib.Nifti1Image(orig_flair, affine=np.eye(4))
        nib.save(nifti_img, f"rotation_exp/{pid}/orig/orig_flair_{pid}.nii.gz")

        orig_seg = no_rotate_batch['segs'][22][i].cpu().numpy()
        nifti_img = nib.Nifti1Image(orig_seg, affine=np.eye(4))
        nib.save(nifti_img, f"rotation_exp/{pid}/orig/orig_seg_{pid}.nii.gz")
        
        for angle_x in angles:
            for angle_y in angles:
                for angle_z in angles:

                    logging.info(f"Starting rotation for subject {pid}: angle_x={angle_x}, angle_y={angle_y}, angle_z={angle_z}")
                    rotated_batch = CalObj(batch, angle_x, angle_y, angle_z)                
                    
                    centered_batch = BatchedCenterObj(rotated_batch)
                    logging.info(f"Completed rotation for subject {pid}: angle_x={angle_x}, angle_y={angle_y}, angle_z={angle_z}")

                    # save rotated, centered
                    t1_post = centered_batch['mris']['t1_post'][i].cpu().numpy()
                    nifti_img = nib.Nifti1Image(t1_post, affine=np.eye(4))
                    nib.save(nifti_img, f"rotation_exp/{pid}/rotated/t1_post_{pid}_{angle_x}_{angle_y}_{angle_z}.nii.gz")

                    flair = centered_batch['mris']['flair'][i].cpu().numpy()
                    nifti_img = nib.Nifti1Image(flair, affine=np.eye(4))
                    nib.save(nifti_img, f"rotation_exp/{pid}/rotated/flair_{pid}_{angle_x}_{angle_y}_{angle_z}.nii.gz")

                    seg = centered_batch['segs'][22][i].cpu().numpy()
                    nifti_img = nib.Nifti1Image(seg, affine=np.eye(4))
                    nib.save(nifti_img, f"rotation_exp/{pid}/rotated/seg_{pid}_{angle_x}_{angle_y}_{angle_z}.nii.gz")

    step += 1
    if step == 2: break