# File: 1a_extract_features.py
# Date: 03/15/2024
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

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
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

#-----------------------------#
#### 1. DIRECTORIES SET UP ####
#-----------------------------#
setup()
MRI_DIR = 'data/preprocessing/output/7_COMPLETED_PREPROCESSED'
SEGS_DIR = 'data/segmentations/'
SEGS_PATHS = [f for f in os.listdir(SEGS_DIR) if f.startswith('Segmentation')]
OUTPUT_DIR = 'data/radiomics/features'
OUTPUT_FILE = f'{OUTPUT_DIR}/features.csv'
LOG_FILE = f'{OUTPUT_DIR}/log.txt'
MODALITIES = ['AX_3D_T1_POST', 'AX_DIFFUSION', 'AX_ADC', 'SAG_3D_FLAIR']
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def extract_features(s, e):
    logging.info(f'\nExtracting features for subject {s}...')
    seg_path = SEGS_DIR + [f for f in SEGS_PATHS if f.startswith(f'Segmentation {s}.nii')][0]
    seg = sitk.ReadImage(seg_path)
    seg_whole_tumor = sitk.BinaryThreshold(seg, lowerThreshold=1, insideValue=22)
    segmentation_labels = [int(v) for v in np.unique(sitk.GetArrayFromImage(seg)) if v != 0]
    if len(segmentation_labels) > 1: segmentation_labels.append(22)
    session = f'{s}_Brainlab'
    if len(lsdir(f'{MRI_DIR}/{s}')) > 1:
        session = lsdir(f'{MRI_DIR}/{s}')[0]
        logging.warning(f'More than one session for {s}, using the first one')

    s_modalities_paths = lsdir(f'{MRI_DIR}/{s}/{session}')    
    s_modalities = [m.split('-')[-1] for m in s_modalities_paths]

    for modality in MODALITIES:
        if modality not in s_modalities:
            logging.warning(f'Missing modality {modality} for subject {s}')
            continue

        modality_path = s_modalities_paths[s_modalities.index(modality)]
        mri_path = f'{MRI_DIR}/{s}/{session}/{modality_path}/{session}_{modality_path}.nii.gz'
        
        for seg_lab in segmentation_labels:
            try:
                if seg_lab == 22:
                    result = e.execute(mri_path, seg_whole_tumor, label=seg_lab)
                else: 
                    result = e.execute(mri_path, seg_path, label=seg_lab)
                result_trimmed = result.copy()
                for k in result.keys():
                    if not isinstance(result[k], np.ndarray):
                        del result_trimmed[k]
                    elif result[k].size > 1:
                        del result_trimmed[k]
                
                features_row = pd.concat([pd.DataFrame({'Subject Number': [s], 'Modality': [modality], 'Segmentation Label': seg_lab}), pd.Series(result_trimmed).to_frame().T], axis=1)
                if not os.path.exists(OUTPUT_FILE):
                    features_row.to_csv(OUTPUT_FILE, index=False)
                else:
                    features_row.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)
            
                logging.info(f"Subject {s} (label {seg_lab}): Success!\n")
            except Exception as _:
                logging.error(f"Subject {s}: FAILED\n")
                continue
def main():
    _, _, labels_df = count_subjects(verbose=False)
    subjects = labels_df['Subject Number'].to_list()
    n = len(subjects)

    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(message)s')
    overall_begin_time = time.time()
    overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = '-' * 80
    logging.info(bar)
    logging.info(f'Starting radiomics feature extraction for n = {n} subjects at {overall_start_time}')

    radiomics.setVerbosity(logging.INFO)
    # handler = logging.FileHandler(LOG_FILE, 'a')
    # formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    # handler.setFormatter(formatter)
    # radiomics.logger.addHandler(handler)

    extractor = featureextractor.RadiomicsFeatureExtractor(correctMask=True)
    extractor.enableAllFeatures()
    # extractor.enableAllImageTypes()
    # logging.info('Pyradiomics extractor used:')
    # logging.info(f'Settings: {pprint(extractor.settings)}')
    # logging.info(f'Enabled features: {pprint(extractor.enabledFeatures)}')
    # logging.info(f'Enabled image types (filters): {pprint(extractor.enabledImagetypes)}')
    logging.info(bar)
    
    for subject in tqdm(subjects):
        extract_features(subject, extractor)
    
    overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_time_elapsed = time.time() - overall_begin_time
    hours, rem = divmod(overall_time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    logging.info(f'Completed radiomics feature extraction for all subjects at {overall_end_time}')
    logging.info(f'Total elapsed time: {overall_time_elapsed}\n')
    logging.info(bar)

if __name__ == '__main__':
    main()