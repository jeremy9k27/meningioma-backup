# File: 2a_convert_dicoms_to_nifti.py
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
# This script converts dicom files to nifti files, 
# taking roughly 15-20min to convert all Brainlab scans as of 1/12/24.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/NURIPS_downloads/Meningiomas_handchecked/*/*_Brainlab/ready_for_preprocessing/*/resources/DICOM/*.dcm
#   * code/preprocessing/dcm2niix
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/2_NIFTI/*/*_Brainlab/*.nii.gz
#   * data/preprocessing/output/2_NIFTI/log.txt

#--------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
import os
from tqdm import tqdm
from datetime import datetime
from utils import lsdir, setup

#---------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
setup()

data_dir = 'data/round2_preprocessing/NURIPS_downloads/Meningiomas_R2'
dcm = 'resources/DICOM'
output_dir = 'data/round2_preprocessing/output/2_NIFTI'
log_file = 'data/round2_preprocessing/output/2_NIFTI/log.txt'
dcm2niix = './code/preprocessing/dcm2niix'

if not os.path.exists(output_dir): os.makedirs(output_dir)

#---------------------------------#
#### 2. CONVERT DICOMS 2 NIFTI ####
#---------------------------------#
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
bar = '-' * 80
os.system(f"echo '\n{bar}\n' >> {log_file}")
os.system(f"echo 'Running script 2a_convert_dicoms_to_nifti.py at {date}\n' >> {log_file}")
print(f"Logging output to {log_file}")

for subject in tqdm(lsdir(data_dir), desc="Subjects"):
    for session in tqdm(lsdir(f'{data_dir}/{subject}'), desc="Sessions", leave=False):
        for scan in tqdm(lsdir(f'{data_dir}/{subject}/{session}/scans'), desc="Scans", leave=False):
            cur_input_dir = f'{data_dir}/{subject}/{session}/scans/{scan}/{dcm}'
            cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
            if not os.path.exists(cur_output_dir): 
                os.makedirs(cur_output_dir)
                log_cmd = f"echo '\nConverting {session}/{scan}' >> {log_file}"
                convert_cmd = f"{dcm2niix} -z y -9 -d -v 0 -o {cur_output_dir} -f {session}_{scan} {cur_input_dir} >> {log_file}" # -f %d
                os.system(log_cmd)
                os.system(convert_cmd)