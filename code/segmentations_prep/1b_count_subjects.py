# File: 1b_count_subjects.py
# Date: 03/14/2024
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

#--------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup, lsdir
import pandas as pd

setup()

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

# %%
methylation_mris, methylation_mris_w_segs, methylation_labels = count_subjects()
print(methylation_labels['MethylationSubgroup'].value_counts())
chr1p_mris, chr1p_mris_w_segs, chr1p_labels = count_subjects(outcome='Chr1p')
print(chr1p_labels['Chr1p'].value_counts())
chr22q_mris, chr22q_mris_w_segs, chr22q_labels = count_subjects(outcome='Chr22q')
print(chr22q_labels['Chr22q'].value_counts())
chr9p_mris, chr9p_mris_w_segs, chr9p_labels = count_subjects(outcome='Chr9p')
print(chr9p_labels['Chr9p'].value_counts())
tert_mris, tert_mris_w_segs, tert_labels = count_subjects(outcome='TERT')
print(tert_labels['TERT'].value_counts())
# %%
