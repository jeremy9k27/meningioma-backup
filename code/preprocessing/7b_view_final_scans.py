# File: 7b_view_final_scans.py
# Date: 02/11/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description:

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to
#
# This script relies on the following file(s) as inputs:
#   *
#   *
#
# This script generates the following file(s) as outputs:
#   *
#   *
#
# Warnings:

#--------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup, lsdir
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import ants

setup()

dir_of_interest = '8_TO_SEGMENT'
data_dir = f'data/preprocessing/output/{dir_of_interest}'
output_dir = f'data/preprocessing/output/THUMBS_{dir_of_interest}'

if not os.path.exists(output_dir): os.makedirs(output_dir)

def visualize_scans(subject='111', orientation='IAL'):
    for session in lsdir(f'{data_dir}/{subject}'):
        if os.path.exists(f'{output_dir}/{session}.png'): continue
        scans = lsdir(f'{data_dir}/{subject}/{session}')
        if len(scans) == 0:
            continue
        fig, axs = plt.subplots(1, len(scans), figsize=(10*len(scans), 12))
        if len(scans) == 1:
            axs = [axs]
        fig.suptitle(f'{session} Final Registration', fontsize=32, y = 0.95)
        for i, scan in enumerate(scans):
            final_path = f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz'
            if not os.path.exists(final_path): continue
            final = ants.image_read(final_path, reorient=orientation)
            if len(final.shape) > 3: continue
            slice = final.shape[0] // 2
            axs[i].imshow(final.numpy()[slice, :, :], cmap='gray')
            axs[i].set_title(f'{scan}', fontsize=24)
            axs[i].set_xlabel(f'Slice: {slice}/{final.shape[0]} for shape {final.shape}', fontsize=18)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        fig.savefig(f'{output_dir}/{session}.png')
        plt.close(fig)

for subject in tqdm(lsdir(data_dir), desc='Visualizing scans'):
    visualize_scans(subject=subject)