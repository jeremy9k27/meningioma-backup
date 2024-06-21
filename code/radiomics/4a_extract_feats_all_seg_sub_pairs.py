# File: 1a_extract_features.py
# Date: 03/15/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Performs radiomics feature extraction on the preprocessed MRI images using their segmentations.
#
# For each subject in the cohort, this script extracts radiomics features from the preprocessed MRI images using their accompanying segmentations.
# See the description of the extract_features function for more details on the extraction process.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/7_COMPLETED_PREPROCESSED/*
#   * data/segmentations/*
#
# This script generates the following file(s) as outputs:
#   * data/radiomics/features6/features.csv
#   * data/radiomics/features6/features_wide.csv
#   * data/radiomics/features6/log.txt

# Package imports
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup, lsdir
from utils import count_subjects
from tqdm import tqdm
import radiomics
from radiomics import featureextractor
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime
import SimpleITK as sitk

# Set up the directories and paths, define global constants
setup()
MRI_DIR = 'data/preprocessing/output/7c_NONLIN_WARP_COMPLETED_PREPROCESSED'
SEGS_DIR = 'data/8_mni_registered_mixed_segs/'
SEGS_PATHS = [f for f in os.listdir(SEGS_DIR) if f.startswith('Segmentation')]
OUTPUT_DIR = 'data/4a_radiomics_for_all_seg_sub_pairs'
OUTPUT_FILE = f'{OUTPUT_DIR}/features.csv'
LOG_FILE = f'{OUTPUT_DIR}/log.txt'
DESIRED_SEQUENCES = ['AX_3D_T1_POST', 'AX_DIFFUSION', 'AX_ADC', 'SAG_3D_FLAIR']
_, SUBJECTS, _ = count_subjects(
    labels_file='data/labels/MeningiomaBiomarkerData.csv',
    mri_dir=MRI_DIR,
    segs_dir=SEGS_DIR,
    drop_by_outcome=False
)
N = len(SUBJECTS)

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(message)s')
OVERALL_BEGIN_TIME = time.time()
OVERALL_START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
BAR = '-' * 80
logging.info(BAR)
logging.info(f'Starting radiomics feature extraction for n = {N} subjects at {OVERALL_START_TIME}')
radiomics.setVerbosity(level=60) # logging.INFO or level=60

# Instantiate the PyRadiomics feature extractor
EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(correctMask=False)
EXTRACTOR.enableAllFeatures()
# extractor.enableAllImageTypes() # uncomment if you want more than the default features extracted

logging.info(BAR)

def assert_images_geometry(sitk_images, paths):
    """
    Helper fn, asserts that all images in the list of SimpleITK.Image objects have the same geometry.

    Parameters
    ----------
    sitk_images : list
        A list of SimpleITK.Image objects.
    paths : list
        A list of paths to the SimpleITK.Image files.

    Returns
    -------
    tuple
        A tuple containing the origin, spacing, and direction of the first image in the list, if all images indeed have the same geometry.
    """
    if not sitk_images: raise ValueError("The list of segmentations is empty.")

    first_image = sitk_images[0]
    first_origin = first_image.GetOrigin()
    first_spacing = first_image.GetSpacing()
    first_direction = first_image.GetDirection()

    if len(sitk_images) == 1:
        return first_origin, first_spacing, first_direction
    
    for i, img in enumerate(sitk_images[1:], start=1):
        if not (img.GetOrigin() == first_origin and
                img.GetSpacing() == first_spacing and
                img.GetDirection() == first_direction):
            raise AssertionError(f"Multiple segmentation files found: {paths}. Segmentation at index {i} has different geometry from the segmentation at index 0. This is a problem!")
    
    return first_origin, first_spacing, first_direction

def get_segs_for_subject(sub_no):
    all_seg_paths = [f for f in SEGS_PATHS if (f.startswith(f'Segmentation {sub_no}.nii') or f.startswith(f'Segmentation {sub_no} '))]
    all_seg_sitks = []
    all_seg_arrays = []
    all_seg_labels = []
    for sp in all_seg_paths:
        seg = sitk.ReadImage(SEGS_DIR + sp)
        all_seg_sitks.append(seg)
        seg_arr = sitk.GetArrayFromImage(seg)
        all_seg_arrays.append(seg_arr)
        all_seg_labels.extend([int(v) for v in np.unique(seg_arr) if v != 0])

    # Check that all segmentations have the same geometry so no funny business goes on in taking the union of the labels
    origin, spacing, direction = assert_images_geometry(all_seg_sitks, all_seg_paths)

    all_seg_labels = sorted(list(set(all_seg_labels)))
    
    # Check to see if subject has enhancing and [(necrotic=3), (resitricted diffusion=6)] segmentations, if so, add appropriate labels (13/16) to the list
    if 1 in all_seg_labels:
        if 3 in all_seg_labels:
            all_seg_labels.append(13)
        if 5 in all_seg_labels:
            all_seg_labels.append(15)
        if 6 in all_seg_labels:
            all_seg_labels.append(16)
            if 5 in all_seg_labels:
                all_seg_labels.append(156)

    all_seg_labels.append(22) # Add the whole tumor mask label
    
    # Create list of masks, one for each segmentation label
    masks = []
    for lab in all_seg_labels:
        mask = np.zeros_like(all_seg_arrays[0])
        for seg_arr in all_seg_arrays:
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
        mask_sitk = sitk.GetImageFromArray(mask)
        mask_sitk.SetOrigin(origin)
        mask_sitk.SetSpacing(spacing)
        mask_sitk.SetDirection(direction)
        masks.append(mask_sitk)
    
    return masks, all_seg_labels

def get_scans_for_subject(sub_no):
    # Subject's first available session
    session = lsdir(f'{MRI_DIR}/{sub_no}')[0]

    # All of the subject's scan names and the corresponding sequences
    available_scan_names = lsdir(f'{MRI_DIR}/{sub_no}/{session}')
    available_scan_sequences = [m.split('-')[-1] for m in available_scan_names]

    # To return
    desired_scan_paths = []
    desired_scan_sequences = []

    # Loop thru the desired sequences and get those that are available from subject's scans
    for sequence in DESIRED_SEQUENCES:
        if sequence in available_scan_sequences:
            desired_scan_sequences.append(sequence)
            scan_name = available_scan_names[available_scan_sequences.index(sequence)]
            desired_scan_paths.append(f'{MRI_DIR}/{sub_no}/{session}/{scan_name}/{session}_{scan_name}.nii.gz')
    
    return desired_scan_paths, desired_scan_sequences

def extract_features(sub_no):
    """
    Given a subject s and a PyRadiomics extractor e, extracts default Pyradiomics features (~101 features) from all present MRI images & segmentation labels of the subject.

    Details
    -------
    * Extraction is completed for each MRI sequence & individual segmentation label present. So for example, if a subject has 3 MRI sequences and 4 segmentation labels, then 12 sets of ~101 features will be extracted.
    * Furthermore, for each sequence feature extraction is performed on the union of all segmentation labels present as well, representing the whole tumor mask (this whole mask is given the label "22" to set it apart).
    * Finally, when possible, for each sequence feature extraction is performed on the union of the following segmentation label subsets [enhancing (1) + necrotic (3) = label 13, enhancing (1) + susceptibility (5) = label 15, enhancing (1) + restricted diffusion (6) = label 16, enhancing (1) + susceptibility (5) + restricted diffusion (6) = label 156].
    * Assumes that the MRI images are stored in the MRI_DIR directory and the segmentations are stored in the SEGS_DIR directory.
    * Saves the extracted features to the OUTPUT_FILE csv file.
    """
    logging.info(f"\nExtracting features from all subjects using subject {sub_no}'s segmentation file...")
    
    # Get the masks and segmentation labels for the subject
    masks, annotation_labels = get_segs_for_subject(sub_no)

    for c_sub in tqdm(SUBJECTS, total=N, dynamic_ncols=True, position=1, desc=f'Loop of subjects providing scans', leave=False, colour='green'):
        scan_paths, scan_sequences = get_scans_for_subject(c_sub)
        for scan, sequence in tqdm(zip(scan_paths, scan_sequences), total=len(scan_paths), dynamic_ncols=True, position=2, desc=f'Scans from subject {c_sub}', leave=False, colour='blue'):
            for mask, annotation_label in tqdm(zip(masks, annotation_labels), total=len(masks), dynamic_ncols=True, position=3, desc=f'Segmentation masks from subject {sub_no}', leave=False, colour='red'):
                try:
                    result = EXTRACTOR.execute(scan, mask, label=annotation_label)
                    result_trimmed = result.copy()
                    for k in result.keys():
                        if not isinstance(result[k], np.ndarray):
                            del result_trimmed[k]
                        elif result[k].size > 1:
                            del result_trimmed[k]
                    
                    features_row = pd.concat([pd.DataFrame({'Subject Providing Segmentation': [sub_no], 'Segmentation Label': [annotation_label], 'Subject Providing Scan': [c_sub], 'Scan Sequence': [sequence]}), pd.Series(result_trimmed).to_frame().T], axis=1)
                    if not os.path.exists(OUTPUT_FILE):
                        features_row.to_csv(OUTPUT_FILE, index=False)
                    else:
                        features_row.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)
                
                    logging.info(f"[Seg subject: {sub_no}, Annotation label: {annotation_label}, Scan subject: {c_sub}, Sequence: {sequence}] was a SUCCESS!\n")
                except Exception as e:
                    logging.info(f"[Seg subject: {sub_no}, Annotation label: {annotation_label}, Scan subject: {c_sub}, Sequence: {sequence}] FAILED and had an ERROR: {e}\n")
                    continue

def main():    
    # PERFORM FEATURE EXTRACTION PER SUBJECT!!!
    for subject in tqdm(SUBJECTS, total=N, dynamic_ncols=True, position=0, desc='Loop of subjects providing segmentations', leave=True):
        extract_features(subject)
    
    # Log the overall time elapsed
    overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_time_elapsed = time.time() - OVERALL_BEGIN_TIME
    hours, rem = divmod(overall_time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    logging.info(f'Completed radiomics feature extraction for all subjects at {overall_end_time}')
    logging.info(f'Total elapsed time: {overall_time_elapsed}\n')
    logging.info(BAR)

if __name__ == '__main__':
    main()
# %%
