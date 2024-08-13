# File: 4a_skullstrip.py
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
#   * "Container ran out of memory, try increasing RAM in Docker preferences" may show up in the logfile for some scans.
#     If this happens, try increasing the memory allocated to Docker Desktop in settings, then rerun this script.
#     This script takes care of skipping over scans that have already been skull stripped, so you can just rerun it.
#
#   * Some scans may look like they were skull stripped based on the log file, but then when you view the output in 3DSlicer 
#     or using the 5b script, nothing shows up, even though the corresponding scan from step 4 is easy to view and looks normal. 
#     I don't understand what's going on here yet... One such scan is 115_Brainlab/21-AX_3D_T1_POST. Need to check for others...

#--------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup, lsdir
from datetime import datetime
from tqdm import tqdm
import time
import shutil
import os
# from concurrent.futures import ProcessPoolExecutor, as_completed

#-------------------------#
#### 1. FILE WRANGLING ####
#-------------------------#
setup()

skull_stripper = './code/preprocessing/synthstrip-docker'

data_dir = 'data/preprocessing/output/7_COMPLETED_PREPROCESSED' # was 3_N4_BIAS_FIELD_CORRECTED
output_dir = 'data/preprocessing/output/8_SKULLSTRIPPED_COMPLETED_PREPROCESSED' # was 4_SKULLSTRIPPED
log_dir = f'{output_dir}/logs'

if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)

def skullstrip_subject(subject):
    log_file = f'{log_dir}/{subject}-log.txt'
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}'):
            cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
            cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
            if not (os.path.exists(f'{cur_output_dir}/{session}_{scan}.nii.gz') and os.path.exists(f'{cur_output_dir}/brain_mask.nii.gz')): 
                # file wrangling and logging
                if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
                os.system(f"echo 'Using SynthStrip to skull strip {session}/{scan}' >> {log_file}")
                shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')

                # skull strip image and save
                os.system(f"{skull_stripper} -i {cur_input_dir}/{session}_{scan}.nii.gz -o {cur_output_dir}/{session}_{scan}.nii.gz -m {cur_output_dir}/brain_mask.nii.gz >> {log_file}")

def main():
    overall_log_file = os.path.join(output_dir, 'log.txt')
    print(f"Logging output to {overall_log_file}")

    overall_begin_time = time.time()
    overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = '-' * 80
    os.system(f"echo '\n{bar}\n' >> {overall_log_file}")
    os.system(f"echo 'Running script 4a_skullstrip.py at {overall_start_time}\n' >> {overall_log_file}")

    #---------------------------#
    #### 2. SKULLSTRIP SCANS ####
    #---------------------------#
    subjects = lsdir(data_dir)
    for subject in tqdm(subjects, desc='Skull stripping'):
        skullstrip_subject(subject)

    overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_time_elapsed = time.time() - overall_begin_time
    hours, rem = divmod(overall_time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    os.system(f"cat {log_dir}/* >> {overall_log_file}")
    os.system(f"echo '\nCompleted skull stripping all subjects at {overall_end_time}' >> {overall_log_file}")
    os.system(f"echo 'Total elapsed time: {overall_time_elapsed}\n' >> {overall_log_file}")
    os.system(f"echo '{bar}\n' >> {overall_log_file}")

if __name__ == '__main__':
    main()