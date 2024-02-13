# File: 6d_view_propogated_registration.py
# Date: 02/10/2024
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
#   * data/preprocessing/output/5a_SKULLSTRIPPED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/6c_REGISTERED/*/*_Brainlab/*/*.nii.gz

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
def view_registration(subject='111', orientation='IAL'):
    before_dir = f'data/preprocessing/output/5_ZSCORE_NORMALIZED/{subject}/{subject}_Brainlab/'
    after_dir = f'data/preprocessing/output/6_REGISTERED/{subject}/{subject}_Brainlab/'
    skullstrip_dir = f'data/preprocessing/output/4_SKULLSTRIPPED/{subject}/{subject}_Brainlab/'

    suffix = 'Affine_registration_to_AX_3D_T1_POST'
    mni_template_path = 'data/preprocessing/output/6_REGISTERED/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii'
    mni_template = ants.image_read(mni_template_path, reorient=orientation)

    scans = lsdir(before_dir)
    scan_types = [scan.split('-')[-1] for scan in scans]

    has_swi = False
    has_dwi = False
    swi_dwi_scans = []
    if 'AX_SWI' in scan_types: 
        has_swi = True
        swi_path = [s for s in scans if s.endswith('AX_SWI')][0]
        swi_dwi_scans.append(swi_path)
    if 'AX_DIFFUSION' in scan_types: 
        has_dwi = True
        dwi_path = [s for s in scans if s.endswith('AX_DIFFUSION')][0]
        swi_dwi_scans.append(dwi_path)

    # reg scans! (not including swi or dwi)
    reg_scans = scans
    if has_swi: reg_scans = [s for s in reg_scans if not s.endswith('AX_SWI')]
    if has_dwi: reg_scans = [s for s in reg_scans if not s.endswith('AX_DIFFUSION')]

    reg_scans = sorted(reg_scans, key=lambda x: 'AX_3D_T1_POST' in x, reverse=False)

    fig, axs = plt.subplots(3, len(reg_scans), figsize=(5*len(reg_scans), 15))

    for i, scan in enumerate(reg_scans):
        before = ants.image_read(f'{before_dir}/{scan}/{subject}_Brainlab_{scan}.nii.gz', reorient=orientation)
        if 'AX_3D_T1_POST' not in scan:
            if os.path.exists(f'{after_dir}/{scan}/{subject}_Brainlab_{scan}_{suffix}.nii.gz'):
                in_between = ants.image_read(f'{after_dir}/{scan}/{subject}_Brainlab_{scan}_{suffix}.nii.gz', reorient=orientation)
            else:
                in_between = None
        after = ants.image_read(f'{after_dir}/{scan}/{subject}_Brainlab_{scan}.nii.gz', reorient=orientation)
        
        slice = before.shape[0] // 2
        axs[0, i].imshow(before.numpy()[slice, :, :], cmap='gray')
        axs[0, i].set_title(f'{scan} Before')
        axs[0, i].set_xlabel(f'Slice: {slice}/{before.shape[0]} for shape {before.shape}')

        if 'AX_3D_T1_POST' not in scan:
            if in_between is not None:
                slice = in_between.shape[0] // 2
                axs[1, i].imshow(in_between.numpy()[slice, :, :], cmap='gray')
                axs[1, i].set_title(f'{scan} Affine')
                axs[1, i].set_xlabel(f'Slice: {slice}/{in_between.shape[0]} for shape {in_between.shape}')
        else:
            slice = mni_template.shape[0] // 2
            axs[1, i].imshow(mni_template.numpy()[slice, :, :], cmap='gray')
            axs[1, i].set_title(f'MNI Template')
            axs[1, i].set_xlabel(f'{mni_template.shape}')
            axs[1, i].set_xlabel(f'Slice: {slice}/{mni_template.shape[0]} for shape {mni_template.shape}')

        slice = after.shape[0] // 2
        axs[2, i].imshow(after.numpy()[slice, :, :], cmap='gray')
        axs[2, i].set_title(f'{scan} Affine')
        axs[2, i].set_xlabel(f'Slice: {slice}/{after.shape[0]} for shape {after.shape}')

        for ax in axs[:, i]:
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.suptitle(f'Subject {subject} Registration', fontsize=24, y = 0.95)
    plt.show()

    # swi and dwi scans now!
    for scan in swi_dwi_scans:
        cur_scans = [s for s in os.listdir(f'{after_dir}/{scan}') if s.endswith('.nii.gz')]
        before_path = f'{before_dir}/{scan}/{subject}_Brainlab_{scan}.nii.gz'
        cur1_path = [s for s in cur_scans if 'Affine_registration_to' in s][0]
        cur2_path = [s for s in cur_scans if 'Affine_propogated_registration' in s][0]
        cur_final = f'{after_dir}/{scan}/{subject}_Brainlab_{scan}.nii.gz'

        reg1_scan_type = cur1_path.split('Affine_registration_to_')[-1]
        reg1_scan_type = reg1_scan_type.split('.nii.gz')[0]
        reg1_scan_path = [s for s in scans if reg1_scan_type in s][0]
        reg1_path = f'{skullstrip_dir}/{reg1_scan_path}/{subject}_Brainlab_{reg1_scan_path}.nii.gz'

        reg2_path = f'{after_dir}/{reg1_scan_path}/{subject}_Brainlab_{reg1_scan_path}_{suffix}.nii.gz'
        
        ax3dt1post_path = [s for s in scans if 'AX_3D_T1_POST' in s][0]
        reg3_path = f'{after_dir}/{ax3dt1post_path}/{subject}_Brainlab_{ax3dt1post_path}.nii.gz'

        paths_to_plot = [before_path, f'{after_dir}/{scan}/{cur1_path}', f'{after_dir}/{scan}/{cur2_path}', cur_final, reg1_path, reg2_path, reg3_path, mni_template_path]

        titles = [p.split('-')[-1].split('.nii')[0].split('/')[-1] for p in paths_to_plot]
        titles[-1] = 'MNI Template'
        titles[0] = 'Before ' + titles[0]
        titles[2] = titles[2] + ' propogated onto ' + scan.split('.')[0].split('-')[-1]
        titles[3] = 'Final registered ' + titles[3]
        titles[4] = 'Skull stripped ' + titles[4] + ' before registration'
        titles[6] = 'MNI registered ' + titles[6]

        fig, axs = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f'Subject {subject}: {scan.split(".")[0]} Registration', fontsize=24, y = 0.95)
        for i, (path, title) in enumerate(zip(paths_to_plot, titles)):
            im = ants.image_read(path, reorient=orientation)
            slice = im.shape[0] // 2
            axs[i//4, i%4].imshow(im.numpy()[slice, :, :], cmap='gray')
            axs[i//4, i%4].set_title(title, fontsize=12)
            axs[i//4, i%4].set_xlabel(f'Slice: {slice}/{im.shape[0]} for shape {im.shape}')
            axs[i//4, i%4].set_xticks([])
            axs[i//4, i%4].set_yticks([])
        plt.show()
    
    # all final scans now
    fig, axs = plt.subplots(1, len(scans), figsize=(6*len(scans), 7))
    fig.suptitle(f'Subject {subject} Final Registration', fontsize=24, y = 0.95)
    for i, scan in enumerate(scans):
        final_path = f'{after_dir}/{scan}/{subject}_Brainlab_{scan}.nii.gz'
        final = ants.image_read(final_path, reorient=orientation)
        slice = final.shape[0] // 2
        axs[i].imshow(final.numpy()[slice, :, :], cmap='gray')
        axs[i].set_title(f'Registered {scan}')
        axs[i].set_xlabel(f'Slice: {slice}/{final.shape[0]} for shape {final.shape}')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
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
