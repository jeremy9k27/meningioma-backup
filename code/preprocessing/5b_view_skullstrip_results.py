# File: 5b_view_skullstrip_results.py
# Date: 01/24/2024
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
# RAS or LPI?
# This script relies on the following file(s) as inputs:
#   *
#   *
#
# This script generates the following file(s) as outputs:
#   *
#   *
#
# Warnings:

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup, lsdir, read_example_mri, rescale_linear
import matplotlib.pyplot as plt
import numpy as np
from ants import image_read
import cv2

setup()

def view_skullstrip_results(data_dir='data/preprocessing/output', scan_type='AX_3D_T1_POST', subjects_to_plot=None, num_subjects=4, cmap='gray', fig_height=6, orientation='RSA'):
    """
    Author: Lawrence Chillrud
    """
    before_dir = f'{data_dir}/4_INTENSITY_STANDARDIZED'
    after_dir = f'{data_dir}/5_SKULLSTRIPPED'
    subjects = lsdir(after_dir)
    subjects = [s for s in subjects if f'{s}_Brainlab' in lsdir(f'{after_dir}/{s}')]

    if subjects_to_plot is None:
      # pick randomly num_subjects from subjects so long as they have a scan of type scan_type
      all_scans = [' '.join(lsdir(f'{after_dir}/{subject}/{subject}_Brainlab')) for subject in subjects]
      subjects = [s for i, s in enumerate(subjects) if scan_type in all_scans[i]]
      subjects_to_plot = sorted(np.random.choice(subjects, num_subjects, replace=False))
    else:
      num_subjects = len(subjects_to_plot)
    
    print("Subjects to plot: ", subjects_to_plot)

    fig, axs = plt.subplots(num_subjects, 4, figsize=(fig_height*3, fig_height*num_subjects))
    fig.suptitle(f'{scan_type}: Skull stripping Before vs. After', fontsize=36, y=1)
    for i, subject in enumerate(subjects_to_plot):
        session = f'{subject}_Brainlab'
        scans = lsdir(f'{after_dir}/{subject}/{session}')
        scan = [s for s in scans if s.endswith(scan_type)][0]
        print(f"Subject {i + 1} / {num_subjects}: ", subject)
        before = read_example_mri(before_dir, subject, session, scan, ants=True, orientation=orientation).numpy()
        after = read_example_mri(after_dir, subject, session, scan, ants=True, orientation=orientation).numpy()
        mask = image_read(f'{after_dir}/{subject}/{session}/{scan}/brain_mask.nii.gz', reorient=orientation).numpy()

        assert before.shape == after.shape == mask.shape

        global_min = min(before.min(), after.min())
        global_max = max(before.max(), after.max())

        axs[i, 0].set_title(f'{subject} Before', fontsize=24)
        im1 = axs[i, 0].imshow(before[before.shape[0]//2, :, :], cmap=cmap, vmin=global_min, vmax=global_max)
        fig.colorbar(im1, ax=axs[i, 0], orientation='vertical', fraction=0.046, pad=0.04)

        axs[i, 1].set_title(f'{subject} After', fontsize=24)
        im2 = axs[i, 1].imshow(after[after.shape[0]//2, :, :], cmap=cmap, vmin=global_min, vmax=global_max)
        fig.colorbar(im2, ax=axs[i, 1], orientation='vertical', fraction=0.046, pad=0.04)

        axs[i, 2].set_title(f'{subject} Mask', fontsize=24)
        im3 = axs[i, 2].imshow(mask[mask.shape[0]//2, :, :], cmap='viridis')
        fig.colorbar(im3, ax=axs[i, 2], orientation='vertical', fraction=0.046, pad=0.04)

        before = rescale_linear(before,0,1)
        mask = rescale_linear(mask,0,1)
        mask = mask.astype(np.uint8)
        arr_rgb = cv2.cvtColor(before[before.shape[0]//2, :, :], cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(mask[mask.shape[0]//2, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0,1,0), 1)
        im4 = axs[i, 3].imshow(arr_with_contours)
        fig.colorbar(im4, ax=axs[i, 3], orientation='vertical', fraction=0.046, pad=0.04)
        axs[i, 3].set_title(f'{subject} Contours', fontsize=24)

    plt.tight_layout()
    plt.show()

view_skullstrip_results()
# %%
