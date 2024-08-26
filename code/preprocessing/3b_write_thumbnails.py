# File: 3b_write_thumbnails.py
# Date: 08/20/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Given the output from SCAN_TYPE_CLEANUP, make thumbnails to finalize manual renaming

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script writes thumbnails for each scan in the SCAN_TYPE_CLEANUP output. 
# The thumbnails are automatically saved to folders with the overall scan type as the name.
# Then users with radiological expertise should inspect the folders visually to ensure all scans have been categorized correctly.
# Any scans that are in the wrong folder can simply be dragged and dropped into the appropriate folder.
# Finally, script 3c will rename the scans based on the finalized thumbnail locations.
#
# This script relies on the following file(s) as inputs:
#   * data/round2_preprocessing/SCAN_TYPE_CLEANUP/automated.csv
#   * data/round2_preprocessing/SCAN_TYPE_CLEANUP/responses/VirginiaHill/20240820_055815_LGC_edits.csv
#
# This script generates the following file(s) as outputs:
#   * data/round2_preprocessing/SCAN_TYPE_CLEANUP/thumbnails/*

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup
import pandas as pd
import os
import warnings
from ants import image_read
import matplotlib.pyplot as plt

setup()

data_dir = 'data/round2_preprocessing/SCAN_TYPE_CLEANUP'

def write_thumbnail(original_scan_name, clean_name, overwrite=False):
    scan_parts = original_scan_name.split('/')
    session = scan_parts[-2]
    scan = scan_parts[-1]
    scan_path = f'{original_scan_name}/{session}_{scan}.nii.gz'

    title = f'{session}_{scan}'
    clean_name_simple = clean_name.replace('AX_', '').replace('COR_', '').replace('SAG_', '')
    thumbnails_dir = f'{data_dir}/thumbnails/{clean_name_simple}'
    if not os.path.exists(thumbnails_dir): os.makedirs(thumbnails_dir)

    im_path = f'{thumbnails_dir}/{title}.png'

    if not overwrite and os.path.exists(im_path):
        return im_path

    if not os.path.exists(scan_path):
        warnings.warn(f'{scan_path} could not be found (likely due to naming inconsistency from NIFTI converter or NURIPS), so a thumbnail will not be generated!')
        return None
    
    scan = image_read(scan_path, reorient='IAL').numpy()
    if len(scan.shape) != 3:
        warnings.warn(f'{scan_path} has {len(scan.shape)} dimension(s), so a thumbnail will not be generated!')
        return None
    
    slice = scan.shape[0] // 2
    plt.imshow(scan[slice, :, :], cmap='gray')
    plt.axis('off')
    plt.title(f'{title}\nSlice {slice}/{scan.shape[0]} w/shape {scan.shape[1:]}')
    plt.savefig(im_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return im_path

#-----------------------#
#### 1. READ IN DATA ####
#-----------------------#
automated_df = pd.read_csv(f'{data_dir}/automated.csv')
manual_df = pd.read_csv(f'{data_dir}/responses/VirginiaHill/20240820_055815_LGC_edits.csv')
automated_df = automated_df.drop(columns=['given_name'])
manual_df = manual_df.rename(columns={'text': 'clean_name', 'image_id': 'id'})
renaming_df = pd.concat([automated_df, manual_df])

#%%---------------------#
#### 2. WRITE THUMBS ####
#-----------------------#
for row in renaming_df.itertuples():
    thumbnail_path = write_thumbnail(row.id, row.clean_name)
# %%
