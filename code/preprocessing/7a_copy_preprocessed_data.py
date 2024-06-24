# File: 7a_copy_preprocessed_data.py
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
from utils import setup, lsdir
import shutil
from tqdm import tqdm
import os

setup()

data_dir = 'data/preprocessing/output/6c_NONLIN_WARP_REGISTERED'
output_dir = 'data/preprocessing/output/7c_NONLIN_WARP_COMPLETED_PREPROCESSED'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for subject in tqdm(lsdir(data_dir), desc='Copying preprocessed data'):
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}'):
            if not os.path.exists(f'{output_dir}/{subject}/{session}/{scan}'):
                os.makedirs(f'{output_dir}/{subject}/{session}/{scan}')
                if os.path.exists(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz'):
                    shutil.copy(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz', f'{output_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz')
                if os.path.exists(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.json'):
                    shutil.copy(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.json', f'{output_dir}/{subject}/{session}/{scan}/{session}_{scan}.json')