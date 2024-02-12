# File: 3c_parallel_n4_correction.py
# Date: 02/12/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: This script is meant to perform N4 bias field correction on the MRI scans.

#%%------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports
# 1. Set up filepaths
# 2. N4 bias field correction

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to perform N4 bias field correction on the MRI scans, parallelizing the process for faster runtime.
# 
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/2_NIFTI/*/*_Brainlab/*/*.nii.gz
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/*/*_Brainlab/*/bias_field.nii.gz (when save_bias_field = True)
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/log.txt
#
# Warnings: 
#   * I use the ANTsPy library to perform the N4 bias field correction. I could've 
#     chosen to use something like SimpleITK, which would've given me finer control 
#     over bias correction, but I went with ANTsPy because it's simpler and is well 
#     trusted in the radiology community (not that SimpleITK isn't well trusted). 
#     For a demonstration of how to use SimpleITK to perform bias correction, see
#     the following link: 
#     https://github.com/Angeluz-07/MRI-preprocessing-techniques/blob/main/notebooks/03_bias_field_correction.ipynb
#   * I do NOT rescale images to [0, 1] or [0, 255] before bias correction, because I 
#     wanted to preserve the original intensity values and it didn't seem like it made 
#     much of a difference to the algorithm. I did check on one example image 
#     (6/6_Brainlab/12-AX_3D_T1_POST) to see if rescaling makes any difference. The
#     algo took roughly the same time (8.9s vs 9.5s not rescaled vs. rescaled to [0, 1]),
#     while the bias corrected scans I got back differed by 0.00395, or 0.4% 
#     (measured using Frobenius norm difference).
#   * Finally, I let the ANTsPy algorithm determine the mask for the bias field, rather than
#     using a mask that I created myself through e.g. Otsu thresholding. It might be worth
#     revisiting this though. 

#--------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
import os
import shutil
import ants
from datetime import datetime
from tqdm import tqdm
from utils import lsdir, setup
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time

#---------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
setup()

data_dir = 'data/preprocessing/output/2_NIFTI'
output_dir = 'data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED'
overall_log_file = 'data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/log.txt'
log_dir = f'{output_dir}/logfiles'

save_bias_field = False
num_workers = 4

if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)

#%%---------------------------------#
#### 2. N4 BIAS FIELD CORRECTION ####
#-----------------------------------#
def n4_correct_subject(subject):
    log_file = os.path.join(log_dir, f'{subject}-log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}'):
            cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
            cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
            if not os.path.exists(cur_output_dir): 
                # file wrangling and logging
                os.makedirs(cur_output_dir)
                logging.info(f'N4 bias correcting {session}/{scan}')
                shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')

                # load image, correct bias field, and save
                image = ants.image_read(f'{cur_input_dir}/{session}_{scan}.nii.gz')
                image_n4 = ants.n4_bias_field_correction(image)
                ants.image_write(image_n4, f'{cur_output_dir}/{session}_{scan}.nii.gz')

                if save_bias_field:
                    n4_field = ants.n4_bias_field_correction(image, return_bias_field=True)
                    ants.image_write(n4_field, f'{cur_output_dir}/bias_field.nii.gz')

def main():
    # start overal logfile
    overall_log_file = os.path.join(output_dir, 'log.txt')
    print(f"Logging output to {overall_log_file}")

    overall_begin_time = time.time()
    overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = '-' * 80
    os.system(f"echo '\n{bar}\n' >> {overall_log_file}")
    os.system(f"echo 'Running script 3c_parallel_n4_correction.py at {overall_start_time}\n' >> {overall_log_file}")
    os.system(f"echo 'Number of workers used: {num_workers}' >> {overall_log_file}")
    os.system(f"echo 'Save bias field = {save_bias_field}\n' >> {overall_log_file}")

    subjects = lsdir(data_dir)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create a list of futures
        futures = [executor.submit(n4_correct_subject, subject) for subject in subjects]
        
        # Initialize tqdm progress bar
        with tqdm(total=len(futures)) as progress_bar:
            for _ in as_completed(futures):
                progress_bar.update(1)
                # Result of the _ can be obtained if needed, e.g., _ = future
                # result = future.result()

    # finish overal logfile
    overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_time_elapsed = time.time() - overall_begin_time
    hours, rem = divmod(overall_time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    os.system(f"cat {log_dir}/* >> {overall_log_file}")
    os.system(f"echo '\nCompleted n4 bias field correction for all subjects at {overall_end_time}' >> {overall_log_file}")
    os.system(f"echo 'Total elapsed time: {overall_time_elapsed}\n' >> {overall_log_file}")
    os.system(f"echo '{bar}\n' >> {overall_log_file}")

if __name__ == '__main__':
    main()