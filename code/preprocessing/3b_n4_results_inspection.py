# File: 3b_n4_results_inspection.py
# Date: 01/23/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Inspects / visualizes the results of N4 bias field correction on the MRI scans manually.

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to inspects / visualize the results of N4 bias field correction on the MRI scans manually.
# Plots are meant to be viewed in VS Code's interactive window, and then saved from there.
# 
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/2_NIFTI/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/*/*_Brainlab/*/*.nii.gz

#%%
import matplotlib.pyplot as plt
from utils import setup, read_example_mri, image_read, explore_3D_array_comparison
from ipywidgets import interact

setup()

def detailed_n4_inspection(data_dir='data/preprocessing/output', subject='6', session='6_Brainlab', scan='12-AX_3D_T1_POST', cmap='nipy_spectral', orientation='IAL'):
    """Authors: Roberto Mena, with modifications by Lawrence Chillrud"""
    arr_before = read_example_mri(f'{data_dir}/2_NIFTI', subject, session, scan, ants=True, orientation=orientation).numpy()
    arr_after = read_example_mri(f'{data_dir}/3_N4_BIAS_FIELD_CORRECTED', subject, session, scan, ants=True, orientation=orientation).numpy()
    arr_bias_field = image_read(f'{data_dir}/3_N4_BIAS_FIELD_CORRECTED/{subject}/{session}/{scan}/bias_field.nii.gz', reorient=orientation).numpy()

    assert arr_after.shape == arr_before.shape
    assert arr_bias_field.shape == arr_before.shape

    def fn(SLICE):
    
        global_min = min(arr_before[SLICE, :, :].min(), arr_after[SLICE, :, :].min())
        global_max = max(arr_before[SLICE, :, :].max(), arr_after[SLICE, :, :].max())

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(10,10))
        fig.suptitle(f'N4 Bias Field Correction: {session}/{scan}', fontsize=18, y=0.70)

        ax1.set_title('Original', fontsize=15)
        im1 = ax1.imshow(arr_before[SLICE, :, :], cmap=cmap, vmin=global_min, vmax=global_max)
        fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

        ax2.set_title('Bias Corrected', fontsize=15)
        im2 = ax2.imshow(arr_after[SLICE, :, :], cmap=cmap, vmin=global_min, vmax=global_max)
        fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)

        ax3.set_title('Bias Field', fontsize=15)
        im3 = ax3.imshow(arr_bias_field[SLICE, :, :], cmap='viridis')
        fig.colorbar(im3, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)

        plt.tight_layout()
  
    interact(fn, SLICE=(0, arr_before.shape[0]-1))

def inspect_n4_correction(data_dir='data/preprocessing/output', subject='6', session='6_Brainlab', scan='12-AX_3D_T1_POST', cmap='nipy_spectral', orientation='IAL'):
    """Author: Lawrence Chillrud"""
    before = read_example_mri(f'{data_dir}/2_NIFTI', subject, session, scan, ants=True, reorient=orientation)
    after = read_example_mri(f'{data_dir}/3_N4_BIAS_FIELD_CORRECTED', subject, session, scan, ants=True, reorient=orientation)
    explore_3D_array_comparison(before.numpy(), after.numpy(), cmap=cmap, title=f'N4 Bias Field Correction: {session}/{scan}', reorient=orientation)

detailed_n4_inspection()
# %%
