# File: 1e_check_scan_names.py
# Date: 01/04/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Loops thru all scans, returning list of all unique scan names

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports
# 1. Set up filepaths

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script loops thru all scans, returning list of all unique scan names
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/NURIPS_downloads/Meningiomas_handchecked/*/*_Brainlab/ready_for_preprocessing/*/resources/DICOM/*.dcm

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from pprint import pprint
from utils import setup, get_scan_dict, lsdir
import os

#---------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
setup()

# data_dir = 'data/preprocessing/NURIPS_downloads/Meningiomas_handchecked'
data_dir = 'data/preprocessing/output/2_NIFTI'
# dir_of_interest = 'ready_for_preprocessing' # or 'ask_virginia'
dir_of_interest = ''

scan_counts = get_scan_dict(data_dir, dir_of_interest)
pprint(scan_counts)

# %%
has_pre = 0
has_flair = 0
has_neither = 0
scan_of_interest = 'AX_DIFFUSION'
n = 0
for subject in lsdir(data_dir):
    for session in lsdir(os.path.join(data_dir, subject)):
        scans = lsdir(os.path.join(data_dir, subject, session))
        scan_types = [s.split('-')[-1] for s in scans]
        if scan_of_interest in scan_types:
            n += 1
            if 'AX_3D_T1_PRE' in scan_types:
                has_pre += 1
            if 'SAG_3D_FLAIR' in scan_types:
                has_flair += 1
            if 'AX_3D_T1_PRE' not in scan_types and 'SAG_3D_FLAIR' not in scan_types:
                has_neither += 1
                print(sorted(scan_types))

print(f'scan_of_interest (n = {n}):', scan_of_interest)
print(f'has_pre: {has_pre}')
print(f'has_flair: {has_flair}')
print(f'has_neither: {has_neither}')
# %%
