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
# 2. Loop thru for scans

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
import os
from pprint import pprint

def lsdir(path):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

#---------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
if not os.getcwd().endswith('Meningioma'): os.chdir('../..')
if not os.getcwd().endswith('Meningioma'): 
    raise Exception('Please run this script from the Menigioma directory')

data_dir = 'data/preprocessing/NURIPS_downloads/Meningiomas_handchecked'
# data_dir = 'data/preprocessing/output/2_NIFTI'
dir_of_interest = 'ready_for_preprocessing' # or 'ask_virginia'
# dir_of_interest = ''

#------------------------------#
#### 2. LOOP THRU FOR SCANS ####
#------------------------------#
scan_counts = {}
for subject in lsdir(data_dir):
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}/{dir_of_interest}'):
            scan_type = scan.split('-')[1]
            if scan_type in scan_counts:
                scan_counts[scan_type] += 1
            else:
                scan_counts[scan_type] = 1

print(f'Scan counts in {dir_of_interest}:')
pprint(scan_counts)

# %%
