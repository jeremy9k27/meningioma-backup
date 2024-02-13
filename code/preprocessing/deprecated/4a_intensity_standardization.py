# File: 4a_intensity_standardization.py
# Date: 01/22/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: This script is meant to perform intensity standardization on the MRI scans.

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports
# 1. File wrangling & reading
# 2. Normalize images

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to perform intensity standardization on the MRI scans.
# We want to standardize the MRI intensities so that the same scan type across 
# all subjects in our cohort have similar intensity distributions. 
# We will use the Nyul method to standardize intensities.
# We use the intensity_normalization package by Jacob Reinhold to perform the Nyul method.
# As of 1/22/24, this script takes a little over 20min to run on my laptop.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/*/*_Brainlab/*/*.nii.gz
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/4_INTENSITY_STANDARDIZED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/4_INTENSITY_STANDARDIZED/log.txt
#   * data/preprocessing/output/4_INTENSITY_STANDARDIZED/standard_histogram-*.npy
#   * data/preprocessing/output/4_INTENSITY_STANDARDIZED/norm_hist_before-*.png
#   * data/preprocessing/output/4_INTENSITY_STANDARDIZED/norm_hist_after-*.png
#
# Warnings: 103_Brainlab seems to have a problem across the board with all its scans...
#   * scipy/interpolate/_interpolate.py:710: RuntimeWarning: divide by zero encountered in divide
#   * scipy/interpolate/_interpolate.py:713: RuntimeWarning: invalid value encountered in multiply

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup, get_scan_dict, lsdir
from intensity_normalization.normalize.nyul import NyulNormalize
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

setup()
begin_time = time.time()

#-----------------------------------#
#### 1. FILE WRANGLING & READING ####
#-----------------------------------#
data_dir = 'data/preprocessing/output/4ALT_SKULLSTRIPPED' # 'data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED'
output_dir = 'data/preprocessing/output/5ALT_INTENSITY_STANDARDIZED' # 'data/preprocessing/output/4_INTENSITY_STANDARDIZED'
log_file = os.path.join(output_dir, 'log.txt')

if not os.path.exists(output_dir): os.makedirs(output_dir)

scan_types_dict = get_scan_dict(data_dir=data_dir, dir_of_interest='')
scan_types_to_normalize = [scan_type for scan_type in scan_types_dict.keys() if scan_types_dict[scan_type] > 1]

input_filepaths = []
output_filepaths = []

already_normalized = {} # key: scan type, value: count of how many scans of this type have already been normalized
for scan_type in scan_types_to_normalize: already_normalized[scan_type] = 0

# start logfile
print(f"Logging output to {log_file}")
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
bar = '=' * 160
os.system(f"echo '\n{bar}' >> {log_file}")
os.system(f"echo '{bar}' >> {log_file}")
os.system(f"echo '\nRunning ALT script 4a_intensity_standardization.py at {date}\n' >> {log_file}") # remove ALT

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
                
                # if a scan type is NOT in scan_types_to_normalize, that means we only have 1 scan of this type.
                # in this case, we don't need to normalize it, so we just copy it over so we don't have to hunt it down later.
                if scan_type not in scan_types_to_normalize:
                    os.system(f"echo '{session}_{scan}.nii.gz: Copying over since it is the only one of its kind (no Nyul histogram matching possible)\n' >> {log_file}")
                    shutil.copy(f'{cur_input_dir}/{session}_{scan}.nii.gz', f'{cur_output_dir}/{session}_{scan}.nii.gz')
            else:
                already_normalized[scan_type] += 1
            
            # populate input and output filepaths
            input_filepaths.append(f'{data_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz')
            output_filepaths.append(f'{output_dir}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz')

# remove scan types that have already been normalized from the list of scans to normalize
final_scan_types_to_normalize = [scan_type for scan_type in scan_types_to_normalize if already_normalized[scan_type] != scan_types_dict[scan_type]]
for scan_type in scan_types_dict.copy().keys():
    if scan_type not in final_scan_types_to_normalize:
        del scan_types_dict[scan_type]

os.system(f"echo 'The following scan types (counts) will be normalized using Nyul histogram matching:\n' >> {log_file}")
pprint(scan_types_dict, stream=open(log_file, 'a'))
os.system(f"echo '' >> {log_file}")

#%%-------------------------#
#### 2. NORMALIZE IMAGES ####
#---------------------------#
for i, scan_type in enumerate(final_scan_types_to_normalize):
    cur_start_time = time.time()
    line = '-' * 160
    os.system(f"echo '{line}\n' >> {log_file}")
    os.system(f"echo 'Normalizing {scan_type} scans (set {i + 1} / {len(final_scan_types_to_normalize)}) using Nyul histogram matching...\n' >> {log_file}")
    
    print(f"\n\nNormalizing {scan_type} scans (set {i + 1} / {len(final_scan_types_to_normalize)}) using Nyul histogram matching...")
    cur_input_filepaths = [input_fp for input_fp in input_filepaths if input_fp.endswith(f'{scan_type}.nii.gz')]
    cur_output_filepaths = [output_fp for output_fp in output_filepaths if output_fp.endswith(f'{scan_type}.nii.gz')]
    nib_images = [nib.load(input_fp) for input_fp in tqdm(cur_input_filepaths, desc=f"Reading {scan_type} images", leave=True)]
    arr_images = [im.get_fdata() for im in tqdm(nib_images, desc=f"Extracting numpy arrays from {scan_type} images", leave=True)]
    affine_mats = [im.affine for im in tqdm(nib_images, desc=f"Getting affine matrices for {scan_type} images", leave=True)]

    print("Instantiating NyulNormalize...")
    nyul_normalizer = NyulNormalize()
    print("Fitting NyulNormalize...")
    nyul_normalizer.fit(arr_images)
    arr_normalized = []
    for image, output_fp, aff_mat in tqdm(zip(arr_images, cur_output_filepaths, affine_mats), desc=f"Normalizing and saving {scan_type} images", leave=True, total=len(cur_output_filepaths)):
        os.system(f"echo 'Normalizing and saving {output_fp.split('/')[-1]}' >> {log_file}")
        norm_im = nyul_normalizer(image)
        arr_normalized.append(norm_im)
        nib.save(nib.Nifti1Image(norm_im, affine=aff_mat), output_fp)
    
    # standard histogram numpy array save
    os.system(f"echo 'Saving standard histogram for {scan_type}...' >> {log_file}")
    print(f"Saving standard histogram for {scan_type}...")
    nyul_normalizer.save_standard_histogram(f"{output_dir}/standard_histogram-{scan_type}.npy")

    # save before and after histograms for visual validation
    os.system(f"echo 'Done! Now saving before and after histograms for visual validation...' >> {log_file}")
    print("Done! Now saving before and after histograms for visual validation...")
    
    # before
    hp = HistogramPlotter(title=f"N4 corrected {scan_type} (n = {len(cur_input_filepaths)})")
    _ = hp(arr_images, masks=[None] * len(arr_images))
    plt.savefig(f'{output_dir}/norm_hist_before-{scan_type}.png')
    plt.close()

    # after
    hp = HistogramPlotter(title=f"NORMALIZED N4 corrected {scan_type} (n = {len(cur_output_filepaths)}))")
    _ = hp(arr_normalized, masks=[None] * len(arr_normalized))
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

os.system(f"echo 'Completed ALT 4a_intensity_standardization.py at {date}' >> {log_file}") # remove ALT
os.system(f"echo 'Total elapsed time: {total_elapsed_time}' >> {log_file}")
os.system(f"echo '\n{bar}' >> {log_file}")
os.system(f"echo '{bar}' >> {log_file}")
