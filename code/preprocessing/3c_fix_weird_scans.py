# File: 3c_fix_weird_scans.py
# Date: 02/11/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: This script is meant to fix the weird scans that have more than 3 dimensions.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/5/5_Brainlab/2-AX_DIFFUSION/5_Brainlab_2-AX_DIFFUSION.nii.gz
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/5/5_Brainlab/2-AX_DIFFUSION/5_Brainlab_2-AX_DIFFUSION.nii.gz
#
# Warnings: This script can only be ran once! Lawrence ran this on 2/12/24 at 11:28pm CT.

# %%
import ants
from utils import setup, explore_3D_array_comparison, lsdir
import numpy as np

setup()

data_dir = 'data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED'
subject = '5'
session = '5_Brainlab'
scan = '2-AX_DIFFUSION'
scan_path = f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz'
mri = ants.image_read(scan_path, reorient='IAL')
assert len(mri.shape) == 4, f'Expected 4D MRI, but got {mri.shape}'
one = mri[:, :, :, 0]
two = mri[:, :, :, 1]

# explore_3D_array_comparison(one, two)

# %%
# for subject in lsdir(data_dir):
#     for session in lsdir(f'{data_dir}/{subject}'):
#         for scan in lsdir(f'{data_dir}/{subject}/{session}'):
#             scan_type = scan.split('-')[-1]
#             if scan_type == 'AX_DIFFUSION':
#                 scan_path = f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz'
#                 mri = ants.image_read(scan_path, reorient='IAL')
#                 print(f'{session}')
#                 print(mri.direction)
#                 print()

# %%
diffusion_direction = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
ants.from_numpy(data=two, origin=mri.origin[:-1], spacing=mri.spacing[:-1], direction=diffusion_direction).to_file(scan_path)

mri = ants.image_read(scan_path, reorient='IAL')
assert len(mri.shape) == 3

# %%
