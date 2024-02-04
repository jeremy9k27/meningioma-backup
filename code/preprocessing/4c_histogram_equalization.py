# File: 4c_histogram_equalization.py
# Date: 02/02/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: This script is meant to perform histogram equalization on the MRI scans,
# as an alternative to the Nyul method for intensity standardization.

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to perform histogram equalization on the MRI scans.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/*/*_Brainlab/*/*.nii.gz
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/4c_HISTOGRAM_EQUALIZED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/4c_HISTOGRAM_EQUALIZED/log.txt
#   * data/preprocessing/output/4c_HISTOGRAM_EQUALIZED/norm_hist_before-*.png
#   * data/preprocessing/output/4c_HISTOGRAM_EQUALIZED/norm_hist_after-*.png

#--------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup, get_scan_dict, lsdir
from intensity_normalization.plot.histogram import HistogramPlotter
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from pprint import pprint
import nibabel as nib
import logging
from skimage import exposure

setup()
begin_time = time.time()

#-----------------------------------#
#### 1. FILE WRANGLING & READING ####
#-----------------------------------#
data_dir = 'data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED'
output_dir = 'data/preprocessing/output/4c_HISTOGRAM_EQUALIZED'
log_file = os.path.join(output_dir, 'log.txt')

if not os.path.exists(output_dir): os.makedirs(output_dir)

scan_types_dict = get_scan_dict(data_dir=data_dir, dir_of_interest='')

input_filepaths = []
output_filepaths = []

# start logfile
print(f"Logging output to {log_file}")
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
bar = '=' * 160
os.system(f"echo '\n{bar}' >> {log_file}")
os.system(f"echo '{bar}' >> {log_file}")
os.system(f"echo '\nRunning script 4c_histogram_equalization.py at {date}\n' >> {log_file}")

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logging.captureWarnings(True)

# populate input and output filepaths and set up output directories
for subject in lsdir(data_dir):
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}'):

            # set up output directories
            cur_input_dir = f'{data_dir}/{subject}/{session}/{scan}'
            cur_output_dir = f'{output_dir}/{subject}/{session}/{scan}'
            scan_type = scan.split('-')[1]
            if not os.path.exists(cur_output_dir): 
                os.makedirs(cur_output_dir)
                
                # copy the metadata over so we don't need to go hunting for it later
                shutil.copy(f'{cur_input_dir}/{session}_{scan}.json', f'{cur_output_dir}/{session}_{scan}.json')
                            
            # populate input and output filepaths
            input_filepaths.append(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz')
            output_filepaths.append(f'{output_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz')

os.system(f"echo 'The following scan types (counts) will be equalized using histogram equalization:\n' >> {log_file}")
pprint(scan_types_dict, stream=open(log_file, 'a'))
os.system(f"echo '' >> {log_file}")

#%%-------------------------#
#### 2. NORMALIZE IMAGES ####
#---------------------------#
scan_types_to_normalize = scan_types_dict.keys()
for i, scan_type in enumerate(scan_types_to_normalize):
    cur_start_time = time.time()
    line = '-' * 160
    os.system(f"echo '{line}\n' >> {log_file}")
    os.system(f"echo 'Equalizing {scan_type} scans (set {i + 1} / {len(scan_types_to_normalize)}) using histogram equalization...\n' >> {log_file}")
    
    print(f"\n\nEqualizing {scan_type} scans (set {i + 1} / {len(scan_types_to_normalize)}) using histogram equalization...")
    cur_input_filepaths = [input_fp for input_fp in input_filepaths if input_fp.endswith(f'{scan_type}.nii.gz')]
    cur_output_filepaths = [output_fp for output_fp in output_filepaths if output_fp.endswith(f'{scan_type}.nii.gz')]
    nib_images = [nib.load(input_fp) for input_fp in tqdm(cur_input_filepaths, desc=f"Reading {scan_type} images", leave=True)]
    arr_images = [im.get_fdata() for im in tqdm(nib_images, desc=f"Extracting numpy arrays from {scan_type} images", leave=True)]
    affine_mats = [im.affine for im in tqdm(nib_images, desc=f"Getting affine matrices for {scan_type} images", leave=True)]

    arr_equalized = []
    for image, output_fp, aff_mat in tqdm(zip(arr_images, cur_output_filepaths, affine_mats), desc=f"Equalizing and saving {scan_type} images", leave=True, total=len(cur_output_filepaths)):
        os.system(f"echo 'Equalizing and saving {output_fp.split('/')[-1]}' >> {log_file}")
        equalized_im = exposure.equalize_hist(image, mask=image > image.mean()) # or exposure.equalize_adapthist(image, kernel_size=???, clip_limit=???)
        arr_equalized.append(equalized_im)
        nib.save(nib.Nifti1Image(equalized_im, affine=aff_mat), output_fp)
    
    # save before and after histograms for visual validation
    os.system(f"echo 'Done! Now saving before and after histograms for visual validation...' >> {log_file}")
    print("Done! Now saving before and after histograms for visual validation...")
    
    # before
    hp = HistogramPlotter(title=f"N4 corrected {scan_type} (n = {len(cur_input_filepaths)})")
    _ = hp(arr_images, masks=[None] * len(arr_images))
    plt.savefig(f'{output_dir}/norm_hist_before-{scan_type}.png')
    plt.close()

    # after
    hp = HistogramPlotter(title=f"EQUALIZED N4 corrected {scan_type} (n = {len(cur_output_filepaths)}))")
    _ = hp(arr_equalized, masks=[None] * len(arr_equalized))
    plt.savefig(f'{output_dir}/norm_hist_after-{scan_type}.png')
    plt.close()

    # wrap up this scan type
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = str(timedelta(seconds=time.time() - cur_start_time))
    os.system(f"echo 'Done! Elapsed time for this scan type: {elapsed_time}\n' >> {log_file}")
    print(f"Done! Elapsed time for this scan type: {elapsed_time}\n")

# log results
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
total_elapsed_time = str(timedelta(seconds=time.time() - begin_time))

os.system(f"echo 'Completed 4c_histogram_equalization.py at {date}' >> {log_file}") # remove ALT
os.system(f"echo 'Total elapsed time: {total_elapsed_time}' >> {log_file}")
os.system(f"echo '\n{bar}' >> {log_file}")
os.system(f"echo '{bar}' >> {log_file}")
