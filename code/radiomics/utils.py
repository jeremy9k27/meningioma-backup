# File: utils.py
# Date: 03/15/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description:

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import lsdir
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(features_file='data/radiomics/features4/features_wide.csv', labels_file='data/labels/MeningiomaBiomarkerData.csv', outcome='MethylationSubgroup', test_size=9, seed=42, even_test_split=False):
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)
    labels = labels.dropna(subset=[outcome])
    labels = labels[labels['Subject Number'].isin(features['Subject Number'])]
    data = features.merge(labels, on='Subject Number')
    if not even_test_split:
        train_df, test_df = train_test_split(data, test_size=test_size, random_state=seed, stratify=data[outcome])
    else:
        unique_classes = data[outcome].unique()
        train_dfs = []
        test_dfs = []
        test_size_cls = test_size // len(unique_classes)
        min_test_size = None

        for cls in unique_classes:
            # Separate the dataset by class
            data_cls = data[data[outcome] == cls]
            
            # Split each class separately without stratification
            train_df_cls, test_df_cls = train_test_split(data_cls, test_size=test_size_cls, random_state=seed)
            
            # Append the split dataframes to their respective lists
            train_dfs.append(train_df_cls)
            test_dfs.append(test_df_cls)
            
            # Update min_test_size to ensure balanced test set
            if min_test_size is None or test_df_cls.shape[0] < min_test_size:
                min_test_size = test_df_cls.shape[0]

        # Make sure the test sets for all classes have the same size
        for i in range(len(test_dfs)):
            test_dfs[i] = test_dfs[i].sample(n=min_test_size, random_state=seed)

        # Combine the training and test sets
        train_df = pd.concat(train_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
        test_df = pd.concat(test_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
            
    return train_df, test_df

def count_subjects(labels_file='data/labels/MeningiomaBiomarkerData.csv', mri_dir='data/preprocessing/output/7_COMPLETED_PREPROCESSED', segs_dir='data/segmentations', outcome='MethylationSubgroup', verbose=False, drop_by_outcome=True):
    """
    Given a labels file, MRI directory, segmentations directory, and outcome variable (prediction task), this function returns: 
    * the number of subjects with MRI data & biomarker data;
    * the number of subjects with MRI data + biomarker data + segmentations; 
    * and a dataframe with the labels of the subjects with MRI data, segmentations and labels.

    The outcome variable is only important if drop_by_outcome is True (or if verbose is True).
    
    Parameters
    ----------
    labels_file : str
        The path to the labels file.
    mri_dir : str
        The path to the MRI directory.
    segs_dir : str
        The path to the segmentations directory.
    outcome : str
        The outcome variable of interest. Only used if drop_by_outcome is True.
    verbose : bool
        Whether to print the value counts of the outcome variable.
    drop_by_outcome : bool
        Whether to drop subjects with missing values in the outcome variable of interest (corresponds to True), or just drop subjects who have NaN across all outcomes (corresponds to False). By default, drop_by_outcome=True.
    """
    labels = pd.read_csv(labels_file)
    mri_subjects = lsdir(mri_dir)
    segmentations = [f for f in os.listdir(segs_dir) if f.startswith('Segmentation')]
    seg_subs = list(set([f.split('Segmentation ')[-1].split(' ')[0].split('.nii')[0] for f in segmentations]))

    if drop_by_outcome:
        labels_subs = labels.dropna(subset=[outcome])['Subject Number'].to_list()
    else:
        labels_subs = labels.dropna(how='all')['Subject Number'].to_list()
    
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