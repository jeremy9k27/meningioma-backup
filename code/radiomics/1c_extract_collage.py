# File: 1c_extract_collage.py
# Date: 04/29/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Performs collage feature extraction on the preprocessed MRI images using their segmentations.
#
# For each subject in the cohort, this script extracts collage features from the preprocessed MRI images using their accompanying segmentations.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/7_COMPLETED_PREPROCESSED/*
#   * data/segmentations/*
#
# This script generates the following file(s) as outputs:
#   * data/collage/*

# %% Package imports
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup, lsdir
from utils import count_subjects, write_ndarray
from tqdm import tqdm
import logging
import numpy as np
import time
from datetime import datetime
import SimpleITK as sitk
import collageradiomics

# Set up the directories and paths, define global constants
setup()
MRI_DIR = 'data/preprocessed_mri_scans/7_COMPLETED_PREPROCESSED'
SEGS_DIR = 'data/segmentations/'
SEGS_PATHS = [f for f in os.listdir(SEGS_DIR) if f.startswith('Segmentation')]
OUTPUT_DIR = 'data/collage_sparse_small_windows'
LOG_FILE = f'{OUTPUT_DIR}/log.txt'
MODALITIES = ['AX_3D_T1_POST', 'AX_ADC', 'SAG_3D_FLAIR']

# Collage hyperparameters to search thru
HARALICK_WINDOW_SIZES = [3, 5] # 3, 5, 7, 9, 11
BIN_SIZES = [32, 64] # 10, 15, 20, 25, 30, 64 ??

# %%
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# %%
def get_segs_for_subject(sub_no):
    """
    Given a subject number, gets all available segmentation masks of interest along with their labels

    Parameters
    ----------
    sub_no (int): The subject number
    
    Returns
    -------
    mask_arrays (list): A list of np.ndarrays representing the masks of interest
    seg_labels (list): A list of length len(mask_arrays) containing the unique label present in each mask inside mask_arrays

    Notes
    -----
    * masks of interest include all labels in 1, 2, 3, 4, 5, 6, 13, 15, 156, 22
    * where 1=enhancing, 2=other, 3=necrotic, 4=edema, 5=susceptibility, 6=restricted diffusion, 22=whole tumor mask (union of all 6 labels)
    * 13 = enhancing + necrotic, 15 = enhancing + susceptibility, 156 = enhancing + susceptibility + restricted diffusion
    """
    # Get all available segmentations for the subject and load them
    seg_paths = [f for f in SEGS_PATHS if (f.startswith(f'Segmentation {sub_no}.nii') or f.startswith(f'Segmentation {sub_no} '))]
    seg_arrays = [] # list of segmentations as np.ndarrays
    seg_labels = [] # list of unique labels present in the segmentations
    for sp in seg_paths:
        seg_sitk = sitk.ReadImage(SEGS_DIR + sp)
        seg_arr = sitk.GetArrayFromImage(seg_sitk)
        seg_arrays.append(seg_arr)
        seg_labels.extend([int(v) for v in np.unique(seg_arr) if v != 0])

    seg_labels = sorted(list(set(seg_labels)))
    
    # Construct the hybrid labels of interest (13, 15, 156) if the base labels are present
    if 1 in seg_labels:
        if 3 in seg_labels:
            seg_labels.append(13)
        if 5 in seg_labels:
            seg_labels.append(15)
        if 6 in seg_labels:
            seg_labels.append(16)
            if 5 in seg_labels:
                seg_labels.append(156)

    if len(seg_labels) > 1: seg_labels.append(22) # Add the whole tumor mask label
    
    # Create list of masks, one for each present segmentation label
    mask_arrays = []
    for lab in seg_labels:
        mask = np.zeros_like(seg_arrays[0])
        for seg_arr in seg_arrays:
            if lab == 22:
                mask = np.logical_or(mask > 0, seg_arr > 0)
                mask = mask.astype(int) * 22
            elif lab == 13:
                mask = np.logical_or(mask == 13, seg_arr == 1)
                mask = mask.astype(int) * 13
                mask = np.logical_or(mask == 13, seg_arr == 3)
                mask = mask.astype(int) * 13
            elif lab == 15:
                mask = np.logical_or(mask == 15, seg_arr == 1)
                mask = mask.astype(int) * 15
                mask = np.logical_or(mask == 15, seg_arr == 5)
                mask = mask.astype(int) * 15
            elif lab == 16:
                mask = np.logical_or(mask == 16, seg_arr == 1)
                mask = mask.astype(int) * 16
                mask = np.logical_or(mask == 16, seg_arr == 6)
                mask = mask.astype(int) * 16
            elif lab == 156:
                mask = np.logical_or(mask == 156, seg_arr == 1)
                mask = mask.astype(int) * 156
                mask = np.logical_or(mask == 156, seg_arr == 5)
                mask = mask.astype(int) * 156
                mask = np.logical_or(mask == 156, seg_arr == 6)
                mask = mask.astype(int) * 156
            else:
                mask = np.logical_or(mask == lab, seg_arr == lab)
                mask = mask.astype(int) * lab
        mask_arrays.append(mask)
    
    return mask_arrays, seg_labels

def get_mris_for_subject(sub_no):
    """
    Given a subject number, gets all available MRI modalities for the subject

    Parameters
    ----------
    sub_no (int): The subject number
    
    Returns
    -------
    mri_paths (list): A list of paths to the MRI modalities for the subject
    mri_modalities (list): A list of the MRI modalities present for the subject
    """
    # Get the session name for the subject
    session = f'{sub_no}_Brainlab'
    if len(lsdir(f'{MRI_DIR}/{sub_no}')) > 1:
        session = lsdir(f'{MRI_DIR}/{sub_no}')[0]
        # logging.warning(f'More than one session for {sub_no}, using the first one on file: {session}')

    # Get the MRI modality paths for the subject
    mri_paths = lsdir(f'{MRI_DIR}/{sub_no}/{session}')
    mri_full_paths = [f'{MRI_DIR}/{sub_no}/{session}/{m}/{session}_{m}.nii.gz' for m in mri_paths]
    mri_modalities = [m.split('-')[-1] for m in mri_paths]

    final_paths = []
    final_modalities = []
    for i, found_mod in enumerate(mri_modalities):
        if found_mod in MODALITIES:
            final_paths.append(mri_full_paths[i])
            final_modalities.append(found_mod)
    
    return final_paths, final_modalities

def run_collage(sub_no, window_size, bin_size, c_output_dir):
    """
    Runs the collage feature extraction for a single subject-mask-modality combination

    Parameters
    ----------
    sub_no (int): The subject number
    window_size (int): The window size to use for the Haralick texture extraction
    bin_size (int): The bin size to use for the Haralick texture extraction
    c_output_dir (str): The output directory to save the collage features to
    
    Returns
    -------
    None
    """
    mask_arrays, seg_labels = get_segs_for_subject(sub_no)
    mri_paths, mri_modalities = get_mris_for_subject(sub_no)
    masks = []
    labels = []
    paths = []
    modalities = []
    for i in range(len(seg_labels)):
        for j in range(len(mri_modalities)):
            masks.append(mask_arrays[i])
            labels.append(seg_labels[i])
            paths.append(mri_paths[j])
            modalities.append(mri_modalities[j])
    
    for mask, label, path, modality in tqdm(zip(masks, labels, paths, modalities), desc=f'Subject {sub_no}', total=len(labels), colour='green', position=1, smoothing=0, leave=False, dynamic_ncols=True):
        output_filepath = f'{c_output_dir}/subject-{sub_no}_{modality}_seg-{label}.joblib'
        if os.path.exists(output_filepath):
            logging.info(f'Collage features for subject {sub_no} MRI {modality} label {label} already exist, skipping...')
        else:
            logging.info(f'Extracting collage features for subject {sub_no}, MRI {modality}, segmentation label {label}...')
            try:
                # Load the MRI image, swap axes of mri and mask..!
                mri = sitk.GetArrayFromImage(sitk.ReadImage(path))
                mri = np.swapaxes(mri, 0, 2)
                mask = mask_arrays[i]
                mask = np.swapaxes(mask, 0, 2)

                # Extract the collage features
                collage = collageradiomics.Collage(
                    mri, 
                    mask, 
                    svd_radius=5,
                    haralick_window_size=window_size, 
                    num_unique_angles=bin_size
                )
                collage_features = collage.execute()

                # Save the features to disk
                write_ndarray(collage_features, output_filepath)
                logging.info(f'Collage features for subject {sub_no}, MRI {modality}, segmentation label {label} extracted and saved successfully!')
            
            except Exception as e:
                logging.error(f'Error extracting collage features for subject {sub_no}, MRI {modality}, segmentation label {label}: {str(e)}')
    
    return True

# %%
def main():
    # Get the list of subjects we want to extract features for (those with segmentations and biomarker data available)
    _, _, labels_df = count_subjects(verbose=False, drop_by_outcome=False)
    subjects = labels_df['Subject Number'].to_list()
    n = len(subjects)
    
    # reverse order of subjects list
    subjects = subjects[::-1]

    # Set up logging
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%m/%d %H:%M')
    overall_begin_time = time.time()
    overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = '-' * 80
    logging.info(bar)
    logging.info(f'Starting collage feature extraction for n = {n} subjects at {overall_start_time}')
        
    total_collage_settings = len(HARALICK_WINDOW_SIZES) * len(BIN_SIZES)
    logging.info(f'Total number of Collage hyperparameter settings to be tested: {total_collage_settings}')
    logging.info(bar)

    collage_setting_no = 1
    for bin_size in BIN_SIZES:
        for window_size in HARALICK_WINDOW_SIZES:
            logging.info(f'\nBeginning Collage feature extraction with setting {collage_setting_no}/{total_collage_settings}: using window size {window_size} and bin size {bin_size}')
            begin_time = time.time()
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f'Starting at {start_time}')

            # make output dir if needed
            c_output_dir = f'{OUTPUT_DIR}/windowsize-{window_size}_binsize-{bin_size}'
            if not os.path.exists(c_output_dir): os.makedirs(c_output_dir)
        
            for sub in tqdm(subjects, total=len(subjects), dynamic_ncols=True):
                run_collage(sub, window_size, bin_size, c_output_dir)

            # Log the time elapsed for this setting
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time_elapsed = time.time() - begin_time
            hours, rem = divmod(time_elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
            logging.info(f'Completed collage feat extraction for setting {collage_setting_no}/{total_collage_settings} using window size {window_size} and bin size {bin_size} at {end_time}')
            logging.info(f'Elapsed time: {time_elapsed}\n')
            logging.info(bar)
            collage_setting_no += 1

    # Log the overall time elapsed
    overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_time_elapsed = time.time() - overall_begin_time
    hours, rem = divmod(overall_time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    logging.info(f'\n\nCompleted ALL collage feat extraction for all subjects at {overall_end_time}')
    logging.info(f'Total elapsed time: {overall_time_elapsed}\n')
    logging.info(bar)

if __name__ == '__main__':
    main()