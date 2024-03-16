# File: utils.py
# Date: 03/15/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description:

from preprocessing.utils import lsdir
import pandas as pd
import os

def count_subjects(labels_file='data/labels/MeningiomaBiomarkerData.csv', mri_dir='data/preprocessing/output/7_COMPLETED_PREPROCESSED', segs_dir='data/segmentations', outcome='MethylationSubgroup', verbose=False):
    labels = pd.read_csv(labels_file)
    mri_subjects = lsdir(mri_dir)
    segmentations = [f for f in os.listdir(segs_dir) if f.startswith('Segmentation')]
    seg_subs = list(set([f.split('Segmentation ')[-1].split(' ')[0].split('.nii')[0] for f in segmentations]))

    labels_subs = labels.dropna(subset=[outcome])['Subject Number'].to_list()
    labels_subs = [str(int(s)) for s in labels_subs]

    mris_w_labels = list(set(mri_subjects) & set(labels_subs))
    mris_w_labels_w_segs = list(set(mris_w_labels) & set(seg_subs))

    have = [int(e) for e in mris_w_labels_w_segs]
    have_df = labels[labels["Subject Number"].isin(have)]
    if verbose: print(have_df[outcome].value_counts())

    return sorted(mris_w_labels), sorted(mris_w_labels_w_segs), have_df

def get_subset_scan_counts(subjects, data_dir='data/preprocessing/output/7_COMPLETED_PREPROCESSED'):
    """
    Author: Lawrence Chillrud

    When data_dir is data/preprocessing/output/>=2, then dir_of_interest should be '', 
    otherwise, it should be 'ready_for_preprocessing' or 'ask_virginia'
    """
    scan_counts = {}
    for subject in subjects:
        # for session in lsdir(f'{data_dir}/{subject}'):
        session = lsdir(f'{data_dir}/{subject}')[0]
        for scan in lsdir(f'{data_dir}/{subject}/{session}/'):
            scan_type = scan.split('-')[1]
            if scan_type in scan_counts:
                scan_counts[scan_type] += 1
            else:
                scan_counts[scan_type] = 1
    return scan_counts