# File: 8a_try_fixing_gray.py
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
# %%
from utils import setup, lsdir
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import ants
from ipywidgets import interact
import shutil

setup()

data_dir = f'data/preprocessing/output/7_FINISHED'
output_dir = f'data/preprocessing/output/8_TO_SEGMENT'
if not os.path.exists(output_dir): os.makedirs(output_dir)
mni_template = 'data/preprocessing/output/6_REGISTERED/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii'
mni_scan = ants.image_read(mni_template, reorient='IAL')

delta = 0.001
upper_bound = 0 + delta
lower_bound = 0 - delta

def visualize_scans(subject='111', orientation='IAL'):
    def fn(slice):
        for session in lsdir(f'{data_dir}/{subject}'):
            # if os.path.exists(f'{output_dir}/{session}.png'): continue
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
                # slice = final.shape[0] // 2
                im = final.numpy()[slice, :, :]
                # up = im < upper_bound
                # low = im > lower_bound
                # mni_check = mni_scan[slice, :, :] < mni_scan.mean()
                # mask = up * low * mni_check
                axs[i].imshow(im < im.mean(), cmap='viridis')
                axs[i].set_title(f'{scan}', fontsize=24)
                axs[i].set_xlabel(f'Slice: {slice}/{final.shape[0]} for shape {final.shape}', fontsize=18)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            # fig.savefig(f'{output_dir}/{session}.png')
            # plt.show()
            # plt.close(fig)
    interact(fn, slice=(0, mni_scan.shape[0]-1))

visualize_scans()

# %%
for subject in tqdm(lsdir(data_dir), desc='Subjects'):
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}'):
            if not os.path.exists(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz'):
                print(f'\nWarning: No registered scan found for {session}/{scan}\n')
            else:
                if not os.path.exists(f'{output_dir}/{subject}/{session}/{scan}'):
                    os.makedirs(f'{output_dir}/{subject}/{session}/{scan}')
                    mri = ants.image_read(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz', reorient='IAL')

                    origin = mri.origin
                    spacing = mri.spacing
                    direction = mri.direction

                    fixed_mri = mri.copy()
                    fixed_mri[fixed_mri < fixed_mri.mean()] = fixed_mri.min()
                    # mask = (fixed_mri < upper_bound) * (fixed_mri > lower_bound) * (mni_scan < mni_scan.mean())
                    # fixed_mri[fixed_mri == 0] = fixed_mri.min()

                    ants.from_numpy(fixed_mri.numpy(), origin=origin, spacing=spacing, direction=direction).to_file(f'{output_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz')
                    shutil.copy(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.json', f'{output_dir}/{subject}/{session}/{scan}/{session}_{scan}.json')
# %%
