# File: 6b_view_registration.py
# Date: 02/05/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Visualizes the registration results for manual validation purposes.

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports
# 1. View registration
# 2. Missing AX_3D_T1_POST?
# 3. Missing registered scans?
# 4. Non-1x1x1mm spacing?

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to visualize the registration results for manual validation purposes.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/4c_HISTOGRAM_EQUALIZED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/6_REGISTERED/*/*_Brainlab/*/*.nii.gz

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup, lsdir
import matplotlib.pyplot as plt
import numpy as np
import ants
import os

setup()

#----------------------------#
#### 1. VIEW REGISTRATION ####
#----------------------------#
def view_registration(subject='6', orientation='IAL'):
    before_dir = f'data/preprocessing/output/4c_HISTOGRAM_EQUALIZED/{subject}/{subject}_Brainlab/'
    after_dir = f'data/preprocessing/output/6_REGISTERED/{subject}/{subject}_Brainlab/'
    suffix = 'SyNRA_registration_to_AX_3D_T1_POST'
    mni_template = ants.image_read('data/preprocessing/output/6_REGISTERED/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii', reorient=orientation)

    scans = lsdir(before_dir)
    scans = sorted(scans, key=lambda x: 'AX_3D_T1_POST' in x, reverse=False)

    fig, axs = plt.subplots(3, len(scans), figsize=(5*len(scans), 15))

    for i, scan in enumerate(scans):
        before = ants.image_read(f'{before_dir}/{scan}/{subject}_Brainlab_{scan}.nii.gz', reorient=orientation)
        if 'AX_3D_T1_POST' not in scan:
            in_between = ants.image_read(f'{after_dir}/{scan}/{subject}_Brainlab_{scan}_{suffix}.nii.gz', reorient=orientation)
        after = ants.image_read(f'{after_dir}/{scan}/{subject}_Brainlab_{scan}.nii.gz', reorient=orientation)
        
        slice = before.shape[0] // 2
        axs[0, i].imshow(before.numpy()[slice, :, :], cmap='gray')
        axs[0, i].set_title(f'{scan} Before')
        axs[0, i].set_xlabel(f'Slice: {slice}/{before.shape[0]} for shape {before.shape}')

        if 'AX_3D_T1_POST' not in scan:
            slice = in_between.shape[0] // 2
            axs[1, i].imshow(in_between.numpy()[slice, :, :], cmap='gray')
            axs[1, i].set_title(f'{scan} SyNRA')
            axs[1, i].set_xlabel(f'Slice: {slice}/{in_between.shape[0]} for shape {in_between.shape}')
        else:
            slice = mni_template.shape[0] // 2
            axs[1, i].imshow(mni_template.numpy()[slice, :, :], cmap='gray')
            axs[1, i].set_title(f'MNI Template')
            axs[1, i].set_xlabel(f'{mni_template.shape}')
            axs[1, i].set_xlabel(f'Slice: {slice}/{mni_template.shape[0]} for shape {mni_template.shape}')

        slice = after.shape[0] // 2
        axs[2, i].imshow(after.numpy()[slice, :, :], cmap='gray')
        axs[2, i].set_title(f'{scan} Rigid')
        axs[2, i].set_xlabel(f'Slice: {slice}/{after.shape[0]} for shape {after.shape}')

        for ax in axs[:, i]:
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.suptitle(f'Subject {subject} Registration', fontsize=24, y = 0.95)
    plt.show()

view_registration()

#%%-------------------------------#
#### 2. MISSING AX_3D_T1_POST? ####
#---------------------------------# 
# Find out which subjects/sessions are missing the AX_3D_T1_POST scan, 
# and therefore which subjects/sessions are not yet registered
data_dir = 'data/preprocessing/output/6_REGISTERED'
for subject in lsdir(data_dir):
      for session in lsdir(f'{data_dir}/{subject}'):
        found_ax3dt1post = False
        for scan in lsdir(f'{data_dir}/{subject}/{session}'):
            if 'AX_3D_T1_POST' in scan:
                found_ax3dt1post = True
                break
        if not found_ax3dt1post:
            print(f'Warning: No AX_3D_T1_POST scan found for {session}')

#%%----------------------------------#
#### 3. MISSING REGISTERED SCANS? ####
#------------------------------------#
# Find out which subjects/sessions are missing their registered scans 
# for some reason (error during registration?)
for subject in lsdir(data_dir):
      for session in lsdir(f'{data_dir}/{subject}'):
          for scan in lsdir(f'{data_dir}/{subject}/{session}'):
              if not os.path.exists(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz'):
                  print(f'Warning: No registered scan found for {session}/{scan}')

#%%-----------------------------#
#### 4. NON-1X1X1MM SPACING? ####
#-------------------------------#
# Find out which subjects/sessions have registered scans that are not 1x1x1mm
for subject in lsdir(data_dir):
      for session in lsdir(f'{data_dir}/{subject}'):
          for scan in lsdir(f'{data_dir}/{subject}/{session}'):
              if os.path.exists(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz'):
                  im = ants.image_read(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz')
                  if im.spacing != (1.0, 1.0, 1.0):
                      print(f'Warning: {session}/{scan} does not have 1x1x1mm spacing, it has spacing: {im.spacing}')

# %%
