# File: 3a_n4_bias_field_correction.py
# Date: 01/18/2024
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
# This script is meant to perform N4 bias field correction on the MRI scans.
# As of 1/20/24, this script takes 1h47min to run on my laptop, ~80s/subject.
# 
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/2_NIFTI/*/*_Brainlab/*/*.nii.gz
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/*/*_Brainlab/*/bias_field.nii.gz
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

#---------------------------#
#### 1. SET UP FILEPATHS ####
#---------------------------#
setup()

data_dir = 'data/preprocessing/output/2_NIFTI'
output_dir = 'data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED'
log_file = 'data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/log.txt'

if not os.path.exists(output_dir): os.makedirs(output_dir)

#%%---------------------------------#
#### 2. N4 BIAS FIELD CORRECTION ####
#-----------------------------------#
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
bar = '-' * 80
os.system(f"echo '\n{bar}\n' >> {log_file}")
os.system(f"echo 'Running script 3a_n4_bias_field_correction.py at {date}\n' >> {log_file}")
print(f"Logging output to {log_file}")

for subject in tqdm(lsdir(data_dir), desc="Subjects"):
    for session in tqdm(lsdir(f'{data_dir}/{subject}'), desc="Sessions", leave=False):
        for scan in tqdm(lsdir(f'{data_dir}/{subject}/{session}'), desc="Scans", leave=False):
            cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
            cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
            if not os.path.exists(cur_output_dir): 
                # file wrangling and logging
                os.makedirs(cur_output_dir)
                log_cmd = f"echo '\nN4 bias correcting {session}/{scan}' >> {log_file}"
                os.system(log_cmd)
                shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')

                # load image, correct bias field, and save
                image = ants.image_read(f'{cur_input_dir}/{session}_{scan}.nii.gz')
                image_n4 = ants.n4_bias_field_correction(image)
                n4_field = ants.n4_bias_field_correction(image, return_bias_field=True)
                ants.image_write(image_n4, f'{cur_output_dir}/{session}_{scan}.nii.gz')
                ants.image_write(n4_field, f'{cur_output_dir}/bias_field.nii.gz')
            else:
                print(f'Already bias corrected {session}/{scan}, skipping it...')

date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
os.system(f"echo 'Completed 3a_n4_bias_field_correction.py at {date}\n' >> {log_file}")
os.system(f"echo '\n{bar}\n' >> {log_file}")