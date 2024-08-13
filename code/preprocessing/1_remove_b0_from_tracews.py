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
from tqdm import tqdm

#%%-------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
setup()

data_dir = 'data/round2_preprocessing/NURIPS_downloads/Meningiomas_R2'
alt_dir = 'data/round2_preprocessing/NURIPS_downloads/Meningioma_Removed_B0s'
dcm = 'resources/DICOM'

def check_for_multiple_scan_types(scan_dir):
    scan_types = []
    for f in os.listdir(scan_dir):
        if f.endswith('.dcm'):
            dicom = pydicom.dcmread(f'{scan_dir}/{f}')
            if 'SequenceName' in dicom:
                scan_types.append(dicom.SequenceName.lower())
    return sorted(list(set(scan_types)))

#%%---------------------------#
#### 2. MOVE B0 SCANS AWAY ####
#-----------------------------#
for subject in tqdm(lsdir(data_dir), desc='Subjects', total=len(lsdir(data_dir)), position=0, smoothing=0, dynamic_ncols=True, colour='white'):
    for session in tqdm(lsdir(f'{data_dir}/{subject}'), desc='Sessions', total=len(lsdir(f'{data_dir}/{subject}')), position=1, smoothing=0, dynamic_ncols=True, colour='green', leave=False):
        for scan in tqdm(lsdir(f'{data_dir}/{subject}/{session}/scans'), desc='Scans', total=len(lsdir(f'{data_dir}/{subject}/{session}/scans')), position=2, smoothing=0, dynamic_ncols=True, colour='red', leave=False):
            scan_types_found = check_for_multiple_scan_types(f'{data_dir}/{subject}/{session}/scans/{scan}/{dcm}')
            b0_found = False
            for scan_type in scan_types_found:
                if 'b0' in scan_type:
                    b0_found = True
                    break
            if len(scan_types_found) > 1 and b0_found:
                num = scan.split('-')[0]
                name = scan.split('-')[-1]
                destination_dir = f'{alt_dir}/{subject}/{session}/scans/{num}-Removed_B0_{name}/{dcm}'
                for f in os.listdir(f'{data_dir}/{subject}/{session}/scans/{scan}/{dcm}'):
                    if f.endswith('.dcm'):
                        current_path = f'{data_dir}/{subject}/{session}/scans/{scan}/{dcm}/{f}'
                        dicom = pydicom.dcmread(current_path)
                        if 'SequenceName' in dicom:
                            if 'b0' in dicom.SequenceName.lower():
                                if not os.path.exists(destination_dir): os.makedirs(destination_dir)
                                destination_path = f'{destination_dir}/{f}'
                                os.rename(current_path, destination_path)
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                if os.path.exists(destination_dir): print(f'Moved B0s into {subject}/{session}/scans/{num}-Removed_B0_{name}/')

# %%
# ax_diff_count = 0
# b1000_count = 0
# for subject in tqdm(lsdir(data_dir), desc='Subjects', total=len(lsdir(data_dir)), position=0, smoothing=0, dynamic_ncols=True, colour='white'):
#     for session in lsdir(f'{data_dir}/{subject}'):
#         for scan in lsdir(f'{data_dir}/{subject}/{session}/ready_for_preprocessing'):
#             if scan.endswith('AX_DIFFUSION'):
#                 ax_diff_count += 1
#                 for f in os.listdir(f'{data_dir}/{subject}/{session}/ready_for_preprocessing/{scan}/{dcm}'):
#                     if f.endswith('.dcm'):
#                         current_path = f'{data_dir}/{subject}/{session}/ready_for_preprocessing/{scan}/{dcm}/{f}'
#                         dicom = pydicom.dcmread(current_path)
#                         if 'b1000' in dicom.SequenceName.lower():
#                             b1000_count += 1
#                             break
#                         else:
#                             continue
#                     else:
#                         continue
#             else:
#                 continue

# assert ax_diff_count == b1000_count, f'Found {ax_diff_count} AX_DIFFUSION scans and {b1000_count} b1000 scans.'
# print('Done!')
# %%
