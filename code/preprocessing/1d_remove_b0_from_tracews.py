# File: 1d_remove_b0_from_tracews.py
# Date: 1/4/2023
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: DIFFUSION TRACEW scans have a b0 image at the beginning. 
# This script removes the b0 image from the scan, leaving only the b1000.

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports
# 1. Set up filepaths
# 2. Move b0 scans away

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# DIFFUSION TRACEW scans have a b0 image at the beginning. 
# This script removes the b0 image from the scan, leaving only the b1000.
# Because all DIFFUSION TRACEW scans were renamed to AX_DIFFUSION, just like
# the b1000 scans themselves, we need to search through all AX_DIFFUSION scans
# inside the ready_for_preprocessing dirs of the Meningiomas_handchecked dir.
# 
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/NURIPS_downloads/Meningiomas_handchecked/*/*_Brainlab/ready_for_preprocessing/*-AX_DIFFUSION/resources/DICOM/*.dcm
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/NURIPS_downloads/Meningiomas/*/*_Brainlab/scans/*-B0_FROM_AX_DIFFUSION_TRACEW/resources/DICOM/*.dcm

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
import os
import pydicom
from utils import setup, lsdir

#%%-------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
setup()

data_dir = 'data/preprocessing/NURIPS_downloads/Meningiomas_handchecked'
alt_dir = 'data/preprocessing/NURIPS_downloads/Meningiomas'
dcm = 'resources/DICOM'

#%%---------------------------#
#### 2. MOVE B0 SCANS AWAY ####
#-----------------------------#
for subject in lsdir(data_dir):
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}/ready_for_preprocessing'):
            if scan.endswith('AX_DIFFUSION'):
                num = scan.split('-')[0]
                destination_dir = f'{alt_dir}/{subject}/{session}/scans/{num}-B0_FROM_AX_DIFFUSION_TRACEW/{dcm}'
                for f in os.listdir(f'{data_dir}/{subject}/{session}/ready_for_preprocessing/{scan}/{dcm}'):
                    if f.endswith('.dcm'):
                        current_path = f'{data_dir}/{subject}/{session}/ready_for_preprocessing/{scan}/{dcm}/{f}'
                        dicom = pydicom.dcmread(current_path)
                        if 'b0' in dicom.SequenceName.lower():
                            if not os.path.exists(destination_dir): os.makedirs(destination_dir)
                            destination_path = f'{destination_dir}/{f}'
                            os.rename(current_path, destination_path)
                        else:
                            continue
                    else:
                        continue
                if os.path.exists(destination_dir): print(f'Moved B0s into {subject}/{session}/scans/{num}-B0_FROM_AX_DIFFUSION_TRACEW/')
            else:
                continue

# %%
ax_diff_count = 0
b1000_count = 0
for subject in lsdir(data_dir):
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}/ready_for_preprocessing'):
            if scan.endswith('AX_DIFFUSION'):
                ax_diff_count += 1
                for f in os.listdir(f'{data_dir}/{subject}/{session}/ready_for_preprocessing/{scan}/{dcm}'):
                    if f.endswith('.dcm'):
                        current_path = f'{data_dir}/{subject}/{session}/ready_for_preprocessing/{scan}/{dcm}/{f}'
                        dicom = pydicom.dcmread(current_path)
                        if 'b1000' in dicom.SequenceName.lower():
                            b1000_count += 1
                            break
                        else:
                            continue
                    else:
                        continue
            else:
                continue

assert ax_diff_count == b1000_count, f'Found {ax_diff_count} AX_DIFFUSION scans and {b1000_count} b1000 scans.'
print('Done!')
# %%
