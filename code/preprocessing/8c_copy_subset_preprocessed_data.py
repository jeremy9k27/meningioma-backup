# File: 8c_copy_subset_preprocessed_data.py
# Date: 10/01/2024
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


data_dir = 'data/preprocessing/output/7b_COMPLETED_PREPROCESSED'
output_dir = 'data/preprocessing/output/7b_COMPLETED_PREPROCESSED_SUBSET'
subset = ['AX_3D_T1_POST', 'AX_ADC', 'AX_DIFFUSION', 'SAG_3D_FLAIR']

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for subject in tqdm(lsdir(data_dir), desc='Copying preprocessed data'):
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}'):
            sn = scan.split('-')[-1]
            if sn in subset:
                if not os.path.exists(f'{output_dir}/{subject}/{session}/{scan}'):
                    os.makedirs(f'{output_dir}/{subject}/{session}/{scan}')
                    if os.path.exists(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz'):
                        shutil.copy(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz', f'{output_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz')
                    if os.path.exists(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.json'):
                        shutil.copy(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.json', f'{output_dir}/{subject}/{session}/{scan}/{session}_{scan}.json')