# File: 1a_view_segmentations.py
# Date: 03/14/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: View the segmentations virginia made for the MRI scans in our cohort.

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
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup, explore_3D_array, explore_3D_array_with_mask_contour
import ants
import nibabel as nib
import SimpleITK as sitk
import numpy as np

setup()

segmentations_dir = 'data/segmentations'

fine_files = []
bad_files = []
bad_errors = []
for f in os.listdir(segmentations_dir):
    try:
        # print(f'Trying to read in {f}')
        seg = ants.image_read(f'{segmentations_dir}/{f}')
        # seg = nib.load(f'{segmentations_dir}/{f}')
        # seg = sitk.ReadImage(f'{segmentations_dir}/{f}')
        print(f'Successfully read in {f}, got the following values: {np.unique(seg.numpy())}')
        fine_files.append(f)
    except Exception as e:
        # print('-'*80)
        bad_files.append(f)
        bad_errors.append(e)
        # print(f'{f}: {e}')

# subject = '6'

# seg = ants.image_read(f'{segmentations_dir}/Segmentation {subject}.nii')
# seg = nib.load(f'{segmentations_dir}/Segmentation {subject}.nii')
# seg = sitk.ReadImage(f'{segmentations_dir}/Segmentation {subject}.nii')

# subject = 2
# im = ants.image_read(f'data/preprocessing/output/7_COMPLETED_PREPROCESSED/{subject}/{subject}_Brainlab/2-AX_3D_T1_POST/{subject}_Brainlab_2-AX_3D_T1_POST.nii.gz')
# seg = ants.image_read(f'{segmentations_dir}/Segmentation {subject}.nii.gz')

# explore_3D_array_with_mask_contour(im.numpy(), 1*(seg.numpy() > 0))

# %%
