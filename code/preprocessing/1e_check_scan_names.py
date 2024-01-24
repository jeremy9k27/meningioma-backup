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
from utils import setup, get_scan_dict

#---------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
setup()

data_dir = 'data/preprocessing/NURIPS_downloads/Meningiomas_handchecked'
# data_dir = 'data/preprocessing/output/2_NIFTI'
dir_of_interest = 'ready_for_preprocessing' # or 'ask_virginia'
# dir_of_interest = ''

scan_counts = get_scan_dict(data_dir, dir_of_interest)
pprint(scan_counts)

# %%
