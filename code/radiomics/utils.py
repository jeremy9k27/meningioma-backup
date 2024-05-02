# File: utils.py
# Date: 03/15/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Helper functions for the radiomics arm of the meningioma project.
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import lsdir
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import hashlib

def split_array(array, value):
    # Find the index of the specified value in the array
    index = np.where(array == value)[0][0]
    
    # Create one array with the specified value
    array_with_value = array[index:index+1]
    
    # Create another array with the remaining values
    remaining_array = np.concatenate([array[:index], array[index+1:]])
    
    return remaining_array, array_with_value

# Assuming df is your DataFrame
def create_hash(row):
    # Create a unique string from all row entries, ensuring consistent order
    row_str = ''.join(map(str, row.values))
    # Use hashlib to create a hash from the string
    return hashlib.sha256(row_str.encode()).hexdigest()

def plot_train_test_split(y_train, y_test, output_file=None, class_ids=None):
    """
    Bar graph showing the number of samples per class in the training and testing sets. Called at the end of get_data() below.
    """
    unique_classes = np.unique(np.concatenate((y_train, y_test)))
    train_counts_array = [np.sum(y_train == uc) for uc in unique_classes]
    test_counts_array = [np.sum(y_test == uc) for uc in unique_classes]
    combined_counts_array = pd.DataFrame({'Training': train_counts_array, 'Testing': test_counts_array}, index=unique_classes)
    
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(unique_classes))

    # Plotting the training bars
    train_bars = ax.bar(index - bar_width/2, combined_counts_array['Training'], bar_width, label='Training', color='tab:blue')

    # Plotting the testing bars
    test_bars = ax.bar(index + bar_width/2, combined_counts_array['Testing'], bar_width, label='Testing', color='tab:orange')

    # Adding the bar labels
    ax.bar_label(train_bars, padding=3)
    ax.bar_label(test_bars, padding=3)

    # Setting the rest of the plot
    ax.set_xlabel('Class')
    ax.set_xticks(index)
    ax.set_xticklabels(class_ids if class_ids else unique_classes)
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Train/Test Samples per Class ({len(y_train)}/{len(y_test)} split)')
    ax.legend(title='Dataset')

    plt.tight_layout()

    # Save figure or show it
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()

def plot_data_split(y, title='MethylationSubgroup', output_file=None):
    """
    Bar graph showing the number of samples per class
    """    
    class_counts = pd.Series(y).value_counts()

    if len(np.unique(y)) == 3:
        class_ids = ['Merlin Intact', 'Immune Enriched', 'Hypermetabolic']
    else:
        class_ids = ['Intact', 'Loss']
    
    # Plotting the bar graph of class counts
    plt.figure(figsize=(6, 4))
    bars = sns.barplot(x=class_ids, y=class_counts.values, palette='viridis')
    plt.ylabel('# of samples')
    plt.title(f'{title} (n = {len(y)})')
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.0f'),
                  (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                  ha='center', va='center',
                  size=12,
                  color='white',
                  xytext=(0, 0),
                  textcoords='offset points')

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()

def clean_feature_names(strings):
    """Tidies up the feature names by removing specified substrings and replacing them with ""."""
    replacements = [
        "Mod-AX_3D_", "Mod-SAG_3D_", "_POST", "Mod-AX_", "SegLab-", "Feat-original_"
    ]
    
    # Replace specified substrings with ""
    replaced_strings = []
    for string in strings:
        for r in replacements:
            string = string.replace(r, "")
        # Replace "DIFFUSION" with "DWI"
        string = string.replace("DIFFUSION", "DWI")
        replaced_strings.append(string)
        
    return replaced_strings

def get_data(features_file='data/radiomics/features6/features_wide.csv', labels_file='data/labels/MeningiomaBiomarkerData.csv', outcome='MethylationSubgroup', test_size=9, seed=42, even_test_split=False, scaler_obj=None):
    """
    Prepares training and testing data split for the meningioma project. 
    Implements 0 imputation of NaNs and scales data if scaler_obj is specified (no data leakage during scaling step). 
    Plots bar graph of the split. Prints training/testing feature matrix shapes.

    Parameters
    ----------
    features_file : str
        The path to the features csv file.
    labels_file : str
        The path to the labels csv file.
    outcome : str
        The prediction task variable of interest. By default, outcome='MethylationSubgroup'.
    test_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. By default, test_size=9.
    seed : int
        The random seed for reproducibility. By default, seed=42.
    even_test_split : bool
        Whether the test should should have an even class split. By default, even_test_split=False, in which case the test set has the same class proportionality as the overall dataset.
    scaler_obj : object
        A scaler object to scale the data. By default, scaler_obj=None. E.g., StandardScaler(), MinMaxScaler(), etc.
    
    Returns
    -------
    X_train_df : pandas DataFrame
        The training features matrix.
    y_train : numpy array
        The training labels.
    train_subject_nums : pandas Series
        The training subject numbers.
    X_test_df : pandas DataFrame
        The testing features matrix.
    y_test : numpy array
        The testing labels.
    test_subject_nums : pandas Series
        The testing subject numbers.
    """
    # read in features and labels, merge
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)
    labels = labels.dropna(subset=[outcome])
    labels = labels[labels['Subject Number'].isin(features['Subject Number'])]
    data = features.merge(labels, on='Subject Number')
    data.columns = clean_feature_names(data.columns)
    
    # split data into training and test sets
    if not even_test_split: # preserve class proportions
        train_df, test_df = train_test_split(data, test_size=test_size, random_state=seed, stratify=data[outcome])
    else: # ensure an even class split in the test set
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
    
    # drop columns that are missing all values
    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df[train_df.columns]

    # impute missing values with 0
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # separate features and labels
    train_subject_nums = train_df['Subject Number']
    test_subject_nums = test_df['Subject Number']
    X_train_df = train_df.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
    y_train = train_df[outcome].values.astype(int)
    X_test_df = test_df.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
    y_test = test_df[outcome].values.astype(int)

    # scale data if specified
    if scaler_obj is not None:
        X_train_df = pd.DataFrame(scaler_obj.fit_transform(X_train_df), columns=X_train_df.columns)
        X_test_df = pd.DataFrame(scaler_obj.transform(X_test_df), columns=X_test_df.columns)

    return X_train_df, y_train, train_subject_nums, X_test_df, y_test, test_subject_nums

def prep_data_for_loocv(features_file='data/radiomics/features6/features_wide.csv', labels_file='data/labels/MeningiomaBiomarkerData.csv', outcome='MethylationSubgroup', scaler_obj=None):
    # read in features and labels, merge
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)
    labels = labels.dropna(subset=[outcome])
    labels = labels[labels['Subject Number'].isin(features['Subject Number'])]
    data = features.merge(labels, on='Subject Number')
    data.columns = clean_feature_names(data.columns)
    data = data.dropna(axis=1, how='all').fillna(0)
    X = data.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
    y = data[outcome].values.astype(int)

    # scale data if specified
    if scaler_obj is not None:
        X = pd.DataFrame(scaler_obj.fit_transform(X), columns=X.columns)
    
    # X['subject_ID'] = X.apply(create_hash, axis=1)
    return X, y

def prep_data_for_pca(features_file='data/radiomics/features6/features_wide.csv', labels_file='data/labels/MeningiomaBiomarkerData.csv', outcome='MethylationSubgroup', scaler_obj=None):
    # read in features and labels, merge
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)
    labels = labels.dropna(subset=[outcome])
    labels = labels[labels['Subject Number'].isin(features['Subject Number'])]
    data = features.merge(labels, on='Subject Number')
    data.columns = clean_feature_names(data.columns)
    data = data.dropna(axis=1, how='all').fillna(0)
    X = data.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
    y = data[outcome].values.astype(int)
    subject_ids = data['Subject Number'].values

    # scale data if specified
    if scaler_obj is not None:
        X = pd.DataFrame(scaler_obj.fit_transform(X), columns=X.columns)
    
    constant_feats = [col for col in X.columns if X[col].nunique() == 1]
    X = X.drop(columns=constant_feats)

    X['Subject Number'] = subject_ids
    X[outcome] = y

    return X

def plot_corr_matrix(X, outcome='?', output_dir=None, normalizer=None):
    """Plots the correlation matrix of the radiomics training features."""
    if normalizer is None:
        X_normalized = X
    else:
        X_normalized = pd.DataFrame(normalizer.fit_transform(X), columns=X.columns)
    corr_matrix = X_normalized.corr()
    clean_names = X.columns
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, annot_kws={"size": 12}, fmt=".2f", cmap="viridis", xticklabels=clean_names, yticklabels=clean_names, square=False)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=45, va='top', fontsize=12)
    plt.title(f"{outcome}: Top Radiomics Features Correlation Matrix")
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f'{output_dir}/correlation_matrix.png')
    else:
        plt.show()
    plt.close()

def count_subjects(labels_file='data/labels/MeningiomaBiomarkerData.csv', mri_dir='data/preprocessed_mri_scans/7_COMPLETED_PREPROCESSED', segs_dir='data/segmentations', outcome='MethylationSubgroup', verbose=False, drop_by_outcome=True):
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

def get_subset_scan_counts(subjects, data_dir='data/preprocessed_mri_scans/7_COMPLETED_PREPROCESSED'):
    """
    Returns counts of each scan type located within a folder.

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