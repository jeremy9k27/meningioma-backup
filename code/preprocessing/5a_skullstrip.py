# File: 5a_skullstrip.py
# Date: 01/23/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Performs skull stripping (brain extraction) on the MRI scans in our cohort.

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports
# 1. File wrangling
# 2. Skullstrip scans

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to perform skull stripping with SynthStrip on the MRI scans.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/4_INTENSITY_STANDARDIZED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/4_INTENSITY_STANDARDIZED/*/*_Brainlab/*/*.json
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/5_SKULLSTRIPPED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/5_SKULLSTRIPPED/*/*_Brainlab/*/brain_mask.nii.gz
#   * data/preprocessing/output/5_SKULLSTRIPPED/*/*_Brainlab/*/*.json
#   * data/preprocessing/output/5_SKULLSTRIPPED/log.txt
#
# Warnings:

#--------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup, lsdir
from datetime import datetime
from tqdm import tqdm
import time
import shutil
import os

#-------------------------#
#### 1. FILE WRANGLING ####
#-------------------------#
setup()
begin_time = time.time()

skull_stripper = './code/preprocessing/synthstrip-docker'

data_dir = 'data/preprocessing/output/4_INTENSITY_STANDARDIZED'
output_dir = 'data/preprocessing/output/5_SKULLSTRIPPED'
log_file = os.path.join(output_dir, 'log.txt')

if not os.path.exists(output_dir): os.makedirs(output_dir)

date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
bar = '-' * 80
os.system(f"echo '\n{bar}\n' >> {log_file}")
os.system(f"echo 'Running script 5a_skullstrip.py at {date}\n' >> {log_file}")
print(f"Logging output to {log_file}")

#---------------------------#
#### 2. SKULLSTRIP SCANS ####
#---------------------------#
for subject in tqdm(lsdir(data_dir), desc="Subjects"):
    for session in tqdm(lsdir(f'{data_dir}/{subject}'), desc="Sessions", leave=False):
        for scan in tqdm(lsdir(f'{data_dir}/{subject}/{session}'), desc="Scans", leave=False):
            cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
            cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
            if not os.path.exists(cur_output_dir): 
                # file wrangling and logging
                os.makedirs(cur_output_dir)
                os.system(f"echo 'Using SynthStrip to skull strip {session}/{scan}' >> {log_file}")
                shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')

                # skull strip image and save
                os.system(f"{skull_stripper} -i {cur_input_dir}/{session}_{scan}.nii.gz -o {cur_output_dir}/{session}_{scan}.nii.gz -m {cur_output_dir}/brain_mask.nii.gz >> {log_file}")

time_elapsed = time.time() - begin_time
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
os.system(f"echo 'Completed 3a_n4_bias_field_correction.py at {date}\n' >> {log_file}")
os.system(f"echo 'Total elapsed time: {time_elapsed}' >> {log_file}")
os.system(f"echo '\n{bar}\n' >> {log_file}")