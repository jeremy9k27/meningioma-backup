# %%
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
import joblib
from ants import image_read
import cv2

setup()

# directory handling
data_dir = 'data/plotted_scans_w_top_feats/'
task = 'MethylationSubgroup'
output_dir = f'{data_dir}/{task}'
if not os.path.exists(output_dir): os.makedirs(output_dir)

# Get data prepared
X = prep_data_for_pca(outcome=task, scaler_obj=StandardScaler())
subject_ids = X['Subject Number']
y = X[task]
plot_data_split(y, task)
X = X.drop(columns=['Subject Number', task])
feat_names = X.columns

# %%
def get_nonzero_feats(exp):
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

exp = joblib.load(f'data/lto_fine_lambdas_5-1-24/{task}/exp.pkl')
feats_dict = get_nonzero_feats(exp)
c_coefs = feats_dict['Hypermetabolic']
highlight_names = c_coefs[c_coefs['Cum Var Exp'] <= 0.99].index.to_list()

X_sm = X[highlight_names]
X_sm['label'] = y

plot_corr_matrix(X_sm, outcome=task)

# %%
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
    
    roi_idx = np.where(np.array(seg_labels) == roi)[0][0]
    mask_arr = mask_arrays[roi_idx]

    return mask_arr

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

def plot_mris(arr_list, mask_list, thickness=1, titles=None, suptitle='Suptitle'):
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

    fig.suptitle(suptitle)

    for i, (arr, mask, slice_idx, ax) in enumerate(zip(arr_list, mask_list, max_slice_list, axes.flatten())):
        arr_rgb = cv2.cvtColor(arr[slice_idx, :, :], cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(mask[slice_idx, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0, 255, 0), thickness)
        
        ax.imshow(arr_with_contours)
        ax.axis('off')
        ax.set_title(titles[i] if titles else f'Image {i}')

    plt.show()

def plot_mris_w_feat(feat_of_interest, use_three=True, orientation='IAL', modality=None):
    """
    orientation can be one of: IAL (axial), ASL (coronal), ILS (saggital)
    modality can be one of: T1, FLAIR, DWI, ADC
    """
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
    if use_three:
        subject_nums = [subject_nums[0], subject_nums[len(subject_nums)//2], subject_nums[-1]]
        subject_vals = [subject_vals[0], subject_vals[len(subject_vals)//2], subject_vals[-1]]
    
    masks_of_int = [get_seg_for_subject(sub_no, int(roi), orientation=orientation) for sub_no in subject_nums]
    scans_of_int = [get_mri_for_subject(sub_no, modality, orientation=orientation)[0] for sub_no in subject_nums]
    titles = [f'Subject {sub_no}: {round(val, 3)}' for sub_no, val in zip(subject_nums, subject_vals)]
    plot_mris(scans_of_int, masks_of_int, titles=titles, suptitle=f'{fi}_{pyrad_name}, {modality} {roi_key[roi]}')

# %%
# IAL = axial
# ASL = coronal
for feat_of_interest in highlight_names[:5]:
    try:
        plot_mris_w_feat(feat_of_interest, orientation='IAL', modality='T1')
    except Exception as e:
        print(e)
# %%
