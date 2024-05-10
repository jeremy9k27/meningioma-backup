# %% package imports
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup, lsdir, rescale_linear
from utils import prep_data_for_pca, plot_data_split, plot_corr_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from ants import image_read
import cv2
import warnings

setup()

# directory handling
data_dir = 'results/plotted_scans_w_top_feats/'
task = 'Chr22q' # 'Chr22q'
subtask = 'Merlin Intact' # subtask isn't used when task != 'MethylationSubgroup'
if task == 'MethylationSubgroup':
    output_dir = f'{data_dir}/{task}/{subtask}'
else:
    output_dir = f'{data_dir}/{task}'

if not os.path.exists(output_dir): os.makedirs(output_dir)

# Get data prepared, plot the data split as a bar graph
X = prep_data_for_pca(outcome=task, scaler_obj=StandardScaler())
subject_ids = X['Subject Number']
y = X[task]
plot_data_split(y, task)
X = X.drop(columns=['Subject Number', task])
feat_names = X.columns

def get_nonzero_feats(exp):
    """
    Given an experiment object, returns a dictionary of dataframes containing all the nonzero feature coefficients for each class
    """
    feat_dict = {}
    if exp.n_classes > 2:
        for k, biomarker in enumerate(exp.class_ids):
            coefs = pd.DataFrame(exp.coef[:, k, :], columns=exp.feat_names)
            zero_feats = [col for col in coefs.columns if (coefs[col] == 0.).all()]
            nonzero_coefs = coefs.drop(columns=zero_feats).T
            nonzero_coefs['Absolute Sum'] = nonzero_coefs.abs().sum(axis=1)
            nonzero_coefs = nonzero_coefs.sort_values(by='Absolute Sum', ascending=False)
            nonzero_coefs['Prop Var Exp'] = nonzero_coefs['Absolute Sum'] / nonzero_coefs['Absolute Sum'].sum()                
            nonzero_coefs['Cum Var Exp'] = nonzero_coefs['Prop Var Exp'].cumsum()
            feat_dict[biomarker] = nonzero_coefs
    else:
        coefs = pd.DataFrame(exp.coef.squeeze(), columns=exp.feat_names)
        zero_feats = [col for col in coefs.columns if (coefs[col] == 0.).all()]
        nonzero_coefs = coefs.drop(columns=zero_feats).T
        nonzero_coefs['Absolute Sum'] = nonzero_coefs.abs().sum(axis=1)
        nonzero_coefs = nonzero_coefs.sort_values(by='Absolute Sum', ascending=False)
        nonzero_coefs['Prop Var Exp'] = nonzero_coefs['Absolute Sum'] / nonzero_coefs['Absolute Sum'].sum()
        nonzero_coefs['Cum Var Exp'] = nonzero_coefs['Prop Var Exp'].cumsum()
        feat_dict['Chr22q'] = nonzero_coefs

    return feat_dict

exp = joblib.load(f'results/lto_fine_lambdas_5-1-24/{task}/exp.pkl')
feats_dict = get_nonzero_feats(exp)
if task == 'MethylationSubgroup':
    c_coefs = feats_dict[subtask]
    intercept = exp.intercept[:, exp.class_ids.index(subtask)]
else:
    c_coefs = feats_dict[task]
    intercept = exp.intercept

c_coefs['Avg'] = c_coefs.drop(columns=['Absolute Sum', 'Prop Var Exp', 'Cum Var Exp']).mean(axis=1)
avg_coefs_rounded = round(c_coefs['Avg'], 2)
highlight_names = c_coefs[c_coefs['Cum Var Exp'] <= 0.99].index.to_list()

X_sm = X[highlight_names]
X_sm['label'] = y

plot_corr_matrix(X_sm, outcome=task) # plots the correlation matrix of the nonzero features from the model

y_preds = exp.final_model_dict['test_preds']

# %% Plots heatmap of the nonzero model coefficients
def plot_heatmap(data, task):
    plt.figure(figsize=(max(0.9*data.shape[0], 22), data.shape[0]))  # Adjust the figure size as needed
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Scale'})
    plt.xlabel('Subject in Test Set (Fold)', fontsize=16)
    plt.ylabel('Feature', fontsize=16)
    plt.yticks(rotation=0, fontsize=12)
    plt.title(f'{task} Heatmap of L1-regularized coefs across folds', fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.close()

named_coef_df = c_coefs[c_coefs['Cum Var Exp'] < 0.99].drop(columns=['Absolute Sum', 'Prop Var Exp', 'Cum Var Exp', 'Avg'])
named_coef_df.columns = subject_ids
if task == 'MethylationSubgroup':
    plot_heatmap(named_coef_df, subtask)
else:
    plot_heatmap(named_coef_df, task)

# %%
def construct_model_formula(rounded_coef_series, intercept_value):
    model_formula = ""
    for i in range(len(rounded_coef_series)):
        if i == 0: # first term we don't need to add a plus sign
            model_formula += f'{rounded_coef_series.iloc[i]}({rounded_coef_series.index[i]})'
        else: # all remaining terms we may need to add a plus or minus sign
            if rounded_coef_series.iloc[i] != 0: # if the coefficient is zero, we don't need to add it to the formula
                if rounded_coef_series.iloc[i] > 0: # if the coefficient is positive, we need to add a plus sign
                    model_formula += f' + {np.abs(rounded_coef_series.iloc[i])}({rounded_coef_series.index[i]})'
                else: # if the coefficient is negative, we need to add a minus sign
                    model_formula += f' - {np.abs(rounded_coef_series.iloc[i])}({rounded_coef_series.index[i]})'
            else:
                if intercept_value != 0:
                    if intercept_value > 0:
                        model_formula += f' + {intercept_value}'
                    else:
                        model_formula += f' - {np.abs(intercept_value)}'
                break
    
    return model_formula

# model_formula = construct_model_formula(avg_coefs_rounded, round(intercept.mean(), 2))
# model_name = rf'$y = \sigma({model_formula})$'

roi_key = {
    '1': 'enhancing', 
    '2': 'other', 
    '3': 'necrotic',
    '4': 'edema',
    '5': 'susceptibility', 
    '6': 'restricted diffusion', 
    '22': 'whole tumor',
    '13': 'enhancing + necrotic',
    '15': 'enhancing + susceptibility',
    '156': 'enhancing + susceptibility + restricted diffusion'
}

if task == 'MethylationSubgroup':
    label_key = {
        0: 'Merlin Intact',
        1: 'Immune Enriched',
        2: 'Hypermetabolic'
    }
else:
    label_key = {
        0: 'Intact',
        1: 'Lost'
    }

def get_seg_for_subject(sub_no, roi, orientation='IAL'):
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
    SEGS_PATHS = [f for f in os.listdir('data/segmentations/') if f.startswith('Segmentation')]
    seg_paths = [f for f in SEGS_PATHS if (f.startswith(f'Segmentation {sub_no}.nii') or f.startswith(f'Segmentation {sub_no} '))]
    seg_arrays = [] # list of segmentations as np.ndarrays
    seg_labels = [] # list of unique labels present in the segmentations
    for sp in seg_paths:
        seg_arr = image_read('data/segmentations/' + sp, reorient=orientation).numpy()
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

    seg_labels.append(22) # Add the whole tumor mask label
    
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
    
    roi_idx_search = np.where(np.array(seg_labels) == roi)[0]

    if len(roi_idx_search) == 0:
        roi_idx = -1
        warnings.warn(f'No segmentation found for seg label {roi_key[str(roi)]} in subject {sub_no}, using {roi_key[str(seg_labels[-1])]} mask instead')
    else: 
        roi_idx = roi_idx_search[0]

    mask_arr = mask_arrays[roi_idx]

    return mask_arr, roi_idx

def find_index_with_key(strings, substring):
    """
    Find the index of the first string in a list that contains the given substring.
    
    Args:
    strings (list of str): The list of strings to search.
    substring (str): The substring to look for.

    Returns:
    int: The index of the first string that contains the substring, or -1 if no match is found.
    """
    for index, string in enumerate(strings):
        if substring in string:
            return index
    return -1 

def get_mri_for_subject(sub_no, key, orientation='IAL', MRI_DIR='data/preprocessed_mri_scans/7_COMPLETED_PREPROCESSED'):
    """
    Given a subject number and key, return available MRI scan with name containing key
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
    if key == 'DWI':
        key = 'DIFFUSION'
    path_idx = find_index_with_key(mri_modalities, key)
    if path_idx == -1:
        print(mri_modalities)
        raise ValueError(f'No MRI modality found for {key} in subject {sub_no}')
    
    mri = image_read(mri_full_paths[path_idx], reorient=orientation).numpy()

    return mri, mri_modalities[path_idx]

def get_max_slice(mask):
    return np.argmax(np.sum(mask, axis=(1, 2)))

def plot_mris(arr_list, mask_list, thickness=1, titles=None, suptitle='Suptitle', fpath=None):
    arr_list = [rescale_linear(a, 0, 1) for a in arr_list]
    mask_list = [rescale_linear(m, 0, 1).astype(np.uint8) for m in mask_list]
    max_slice_list = [get_max_slice(m) for m in mask_list]

    num_images = len(arr_list)

    if len(titles) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    if num_images == 1:
        axes = [axes]

    fig.suptitle(suptitle, y = 1.05)

    for i, (arr, mask, slice_idx, ax) in enumerate(zip(arr_list, mask_list, max_slice_list, axes.flatten())):
        arr_rgb = cv2.cvtColor(arr[slice_idx, :, :], cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(mask[slice_idx, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0, 255, 0), thickness)
        
        ax.imshow(arr_with_contours)
        ax.axis('off')
        ax.set_title(titles[i] if titles else f'Image {i}')

    fig.tight_layout()
    plt.savefig(f'{output_dir}/{fpath}', bbox_inches='tight')
    plt.close()

def plot_mris_w_feat(feat_of_interest, feat_num=0, which_three=0, orientation='IAL', modality=None):
    """
    orientation can be one of: IAL (axial), ASL (coronal), ILS (saggital)
    modality can be one of: T1, FLAIR, DWI, ADC
    """
    feat_idx = np.where(c_coefs.index == feat_of_interest)[0][0]
    feat_rank = feat_idx + 1
    feat_var_exp = round(c_coefs['Prop Var Exp'].iloc[feat_idx]*100, 1)

    feat_info = feat_of_interest.split('-')
    if modality is None: 
        modality = feat_info[0]
        fi = modality
    else:
        fi = feat_info[0]
    roi = feat_info[1]
    pyrad_name = feat_info[2]
    sorted_idxs = X[feat_of_interest].argsort()
    subject_by_feat = subject_ids[X[feat_of_interest].iloc[sorted_idxs].index.to_list()]
    feat_vals = X[feat_of_interest].iloc[sorted_idxs]

    # get top three, bottom three and middle three subjects and feat_vals
    top_three = subject_by_feat[-3:]
    top_three_vals = feat_vals[-3:]
    bottom_three = subject_by_feat[:3]
    bottom_three_vals = feat_vals[:3]
    middle_three = subject_by_feat[len(subject_by_feat)//2-1:len(subject_by_feat)//2+2]
    middle_three_vals = feat_vals[len(subject_by_feat)//2-1:len(subject_by_feat)//2+2]

    subject_nums = bottom_three.to_list() + middle_three.to_list() + top_three.to_list()
    subject_vals = bottom_three_vals.to_list() + middle_three_vals.to_list() + top_three_vals.to_list()
    
    if which_three == 0:
        subject_nums = [subject_nums[0], subject_nums[len(subject_nums)//2], subject_nums[-1]]
        subject_vals = [subject_vals[0], subject_vals[len(subject_vals)//2], subject_vals[-1]]
    elif which_three == 1:
        subject_nums = [subject_nums[1], subject_nums[3], subject_nums[7]]
        subject_vals = [subject_vals[1], subject_vals[3], subject_vals[7]]
    else:
        subject_nums = [subject_nums[2], subject_nums[5], subject_nums[6]]
        subject_vals = [subject_vals[2], subject_vals[5], subject_vals[6]]

    masks_of_int = []
    mask_modalities = []
    for sub_no in subject_nums:
        mask_of_int, mask_modality = get_seg_for_subject(sub_no, int(roi), orientation=orientation)
        masks_of_int.append(mask_of_int)
        mask_modalities.append(mask_modality)
    scans_of_int = [get_mri_for_subject(sub_no, modality, orientation=orientation)[0] for sub_no in subject_nums]
    sub_indices = [np.where(subject_ids == sub_no)[0][0] for sub_no in subject_nums]
    titles = [
        f'Subject {sub_no}: {round(val, 3)}\nGround truth label: {label_key[y[sidx]]}\nPredicted label: {label_key[y_preds[sidx]]}' 
        if mask_modality != -1 else f'Subject {sub_no}*: {round(val, 3)}\nGround truth label: {label_key[y[sidx]]}\nPredicted label: {label_key[y_preds[sidx]]}'
        for sub_no, val, sidx, mask_modality in zip(subject_nums, subject_vals, sub_indices, mask_modalities)
    ]

    if task == 'MethylationSubgroup':
        suptitle = f'Task: Methylation Subgroup, {subtask} OvR\nFeature {feat_rank} explaining {feat_var_exp}% of variance (avg coef: {avg_coefs_rounded[feat_of_interest]}): {pyrad_name} on {fi} {roi_key[roi]}\nAs viewed on: {modality} {roi_key[roi]}'
    else:
        suptitle = f'Task: {task}\nFeature {feat_rank} explaining {feat_var_exp}% of variance (avg coef: {avg_coefs_rounded[feat_of_interest]}): {pyrad_name} on {fi} {roi_key[roi]}\nAs viewed on: {modality} {roi_key[roi]}'
    
    if -1 in mask_modalities:
        suptitle += f'\nNote: * indicates {roi_key[roi]} mask unavailable for that subject, in which case the whole tumor mask is displayed instead'
    plot_mris(scans_of_int, masks_of_int, titles=titles, suptitle=suptitle, fpath=f'feat-{feat_num}_set-{which_three}.png')

# %%
# IAL = axial
# ASL = coronal
for i, feat_of_interest in enumerate(highlight_names[:5]):
    plot_mris_w_feat(feat_of_interest, feat_num=i, orientation='IAL', modality='T1', )
    plot_mris_w_feat(feat_of_interest, feat_num=i, which_three=1, orientation='IAL', modality='T1')
    plot_mris_w_feat(feat_of_interest, feat_num=i, which_three=2, orientation='IAL', modality='T1')

# %%
def sub_specific_plot_mri(sub_no):
    sub_idx = np.where(subject_ids == sub_no)[0][0]
    model = round(named_coef_df[sub_no].sort_values(ascending=False, key=abs), 2)
    nonzero_feats = model[model != 0].index.to_list()
    sub_feats = X_sm[nonzero_feats].iloc[sub_idx]
    mf = construct_model_formula(model, round(intercept[sub_idx], 2))
    mri, _ = get_mri_for_subject(sub_no, 'T1')
    mri = rescale_linear(mri, 0, 1)
    seg, _ = get_seg_for_subject(sub_no, roi=22)
    seg = rescale_linear(seg, 0, 1).astype(np.uint8)
    max_slice = get_max_slice(seg)
    arr_rgb = cv2.cvtColor(mri[max_slice, :, :], cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(seg[max_slice, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0, 255, 0), thickness=1)
    
    true_lab = label_key[y[sub_idx]]
    pred_lab = label_key[y_preds[sub_idx]]
    plt.figure(figsize=(8, 8))
    if task == 'MethylationSubgroup':
        title = f'Task: {task} ({subtask})\nSubject {sub_no}\nGround truth: {true_lab}; Predicted: {pred_lab}\nModel Formula:\ny = sigma[{mf}]\nSubject feature values:\n'
    else:
        title = f'Task: {task}\nSubject {sub_no}\nGround truth: {true_lab}; Predicted: {pred_lab}\nModel Formula:\ny = sigma[{mf}]\nSubject feature values:\n'

    for i, (feat, val) in enumerate(sub_feats.items()):
        title += f'{feat}: {round(val, 2)}\n'
    plt.title(title, fontsize=16)
    plt.imshow(arr_with_contours)
    plt.axis('off')
    plt.tight_layout()
    if not os.path.exists(f'{output_dir}/subject_specific_images/'): os.makedirs(f'{output_dir}/subject_specific_images/')
    plt.savefig(f'{output_dir}/subject_specific_images/sub-{sub_no}.png', bbox_inches='tight')
    plt.close()

for subject in subject_ids:
    sub_specific_plot_mri(subject)
