# File: 2_convert_dicoms_to_nifti.py
# Date: 01/08/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: This script converts dicom files to nifti files.

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports
# 1. Set up filepaths
# 2. Convert dicoms to nifti

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/NURIPS_downloads/Meningiomas_handchecked/*/*_Brainlab/ready_for_preprocessing/*/resources/DICOM/*.dcm
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/2_NIFTI/*/*_Brainlab/*.nii.gz

#--------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
import os
from tqdm import tqdm

def lsdir(path):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

#---------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
if not os.getcwd().endswith('Meningioma'): os.chdir('../..')
if not os.getcwd().endswith('Meningioma'): 
    raise Exception('Please run this script from the Menigioma directory')

data_dir = 'data/preprocessing/NURIPS_downloads/Meningiomas_handchecked'
dcm = 'resources/DICOM'
output_dir = 'data/preprocessing/output/2_NIFTI'
log_file = 'data/preprocessing/output/2_NIFTI/log.txt'

if not os.path.exists(output_dir): os.makedirs(output_dir)
#---------------------------------#
#### 2. CONVERT DICOMS 2 NIFTI ####
#---------------------------------#
os.system(f"echo '' > {log_file}")
print(f"Logging output to {log_file}")

# wrap this for loop with a progress bar using tqdm
for subject in tqdm(lsdir(data_dir), desc="Subjects"):
    for session in tqdm(lsdir(f'{data_dir}/{subject}'), desc="Sessions", leave=False):
        for scan in tqdm(lsdir(f'{data_dir}/{subject}/{session}/ready_for_preprocessing'), desc="Scans", leave=False):
            cur_input_dir = f'{data_dir}/{subject}/{session}/ready_for_preprocessing/{scan}/{dcm}'
            cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
            if not os.path.exists(cur_output_dir): 
                os.makedirs(cur_output_dir)
                log_cmd = f"echo '\nConverting {session}/{scan}' >> {log_file}"
                convert_cmd = f"dcm2niix -z y -9 -d -v 0 -o {cur_output_dir} -f %d {cur_input_dir} >> {log_file}"
                os.system(log_cmd)
                os.system(convert_cmd)
            else:
                print(f'Already converted {session}/{scan}, skipping it...')
