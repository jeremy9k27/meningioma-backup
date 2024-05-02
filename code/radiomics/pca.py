# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup, lsdir, explore_3D_array_with_mask_contour
from utils import prep_data_for_pca, plot_data_split, plot_corr_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
import plotly.graph_objects as go

setup()

# directory handling
data_dir = 'data/pca_results/'
task = 'Chr22q'
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
pca = PCA()
principal_components = pca.fit_transform(X)
pca_scores_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
pca_scores_df['Subject Number'] = subject_ids

if not os.path.exists(f"data/pca_results/{task}/pca_scores.csv"): 
    pca_scores_df.to_csv(f"data/pca_results/{task}/pca_scores.csv", index=False)

# %%
# Proportion of Variance Explained
variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)

# Table of Variance Explained
variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(variance_ratio))],
    'Variance Explained': variance_ratio,
    'Cumulative Variance': cumulative_variance
})

variance_df.head(30)  # Display the first few rows

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
c_coefs = feats_dict['Chr22q']
highlight_names = c_coefs[c_coefs['Cum Var Exp'] <= 0.99].index.to_list()

X_sm = X[highlight_names]
X_sm['label'] = y

plot_corr_matrix(X_sm, outcome=task)
# %%
exp = joblib.load(f'data/classic_loo_pca_regression_5-1-24/{task}/exp.pkl')
feats_dict = get_nonzero_feats(exp)
c_coefs = feats_dict['Chr22q']
pcs = c_coefs.index.to_list()[:6]

# given list of strings, strip out the numbers
def strip_numbers(strings):
    return [int(''.join(filter(str.isdigit, s))) for s in strings]

pcs = strip_numbers(pcs)
# %%
feature_type = [feat.split('-')[-1].split('_')[0] for feat in X.columns]
roi_type = [feat.split('-')[1] for feat in X.columns]
modality_type = [feat.split('-')[0] for feat in X.columns]
pyrad_name_type = [feat.split('_')[-1] for feat in X.columns]

def map_categories_to_colors(categories):
    # Get a set of unique categories
    unique_categories = np.unique(categories)
    
    # Get a Brewer color palette from matplotlib (e.g., Set1)
    # Check the length of unique_categories to decide how many colors you need
    num_colors = len(unique_categories)
    color_map = plt.get_cmap('Set1', num_colors)
    
    # Create a dictionary to map categories to colors
    category_to_color = {category: color_map(i) for i, category in enumerate(unique_categories)}
    
    # Map your categories to corresponding colors
    colors = [category_to_color[category] for category in categories]
    
    return colors, category_to_color

# Map the feature types to colors
feature_colors, feature_color_map = map_categories_to_colors(feature_type)

loadings = pca.components_.T

def plot_loadings(highlight_names, n_row=1, n_col=5, PCs=None, ann=True, by_sign=True):
    if PCs is None:
        PCs = np.arange(1, (n_row*n_col) + 1).astype(int)
    
    x_coords = np.linspace(0, len(feat_names), len(feat_names))
    highlight_indices = [np.where(hn == feat_names)[0][0] for hn in highlight_names]

    fig, axes = plt.subplots(n_row, n_col, figsize=(24, 18))
    fig.suptitle(f'{task} PCA Loadings', y = 1.05)
    ax = axes.flatten()

    for i in range(len(ax)):
        var_exp = round(variance_df['Variance Explained'][PCs[i]-1], 2)
        cum_var_exp = round(variance_df['Cumulative Variance'][PCs[i]-1], 2)
        sorted_indices = np.argsort(np.abs(loadings[:, PCs[i]-1]))[::-1]
        if by_sign: 
            colors = ['tab:blue' if x > 0 else 'tab:orange' for x in loadings[sorted_indices, PCs[i]-1]]
        else:
            colors = np.array(feature_colors)[sorted_indices]
        ax[i].scatter(range(len(feat_names)), np.abs(loadings[sorted_indices, PCs[i]-1]), c=colors, alpha=0.1, s=5)
        ax[i].set_title(f'PC{PCs[i]} (Var: {var_exp}, Cum: {cum_var_exp})')
        if i >= (n_row*n_col) - n_col: ax[i].set_xlabel('Features')
        if i % n_row == 0: ax[i].set_ylabel('Absolute Loading Value')

        for j, (index, name) in enumerate(zip(highlight_indices, highlight_names)):
            pos = np.where(sorted_indices == index)[0][0]
            ax[i].scatter(x_coords[pos], np.abs(loadings[sorted_indices[pos], PCs[i]-1]), c=colors[pos], s=50)
            if ann: ax[i].annotate(name, (x_coords[pos], np.abs(loadings[sorted_indices[pos], PCs[i]-1])), textcoords="offset points", xytext=(0,10), ha='left')

    plt.tight_layout()
    plt.show()

plot_loadings(highlight_names, 1, 2)
plot_loadings(highlight_names, 2, 3)
plot_loadings(highlight_names, 2, 3, PCs=pcs)

# %%
# Map classes to colors
color_map = {0: 'red', 1: 'green', 2: 'blue'}
colors = [color_map[cls] for cls in np.arange(len(np.unique(y)))]

# Create biplot for PC1 vs PC2
plt.figure(figsize=(8, 6))
for cls, color in color_map.items():
    idx = [i for i, x in enumerate(y) if x == cls]
    plt.scatter(principal_components[idx, 0], principal_components[idx, 1], c=color, label=cls, alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Biplot (PC1 vs PC2) Colored by Class')
plt.legend(title='Class')
plt.grid(True)
plt.show()

# %%
# Map classes to colors
color_map = {0: 'red', 1: 'green', 2: 'blue'}
colors = [color_map[cls] for cls in np.arange(len(np.unique(y)))]

# Create a 3D plot for PC1 vs PC2 vs PC3
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for cls, color in color_map.items():
    idx = [i for i, x in enumerate(y) if x == cls]
    ax.scatter(principal_components[idx, 0], principal_components[idx, 1], principal_components[idx, 2], c=color, label=cls, alpha=0.7)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Plot (PC1, PC2, PC3) Colored by Class')
ax.legend(title='Class')
plt.show()

# %%

# Map classes to colors
color_map = {0: 'blue', 1: 'red', 2: 'green'}
colors = [color_map[cls] for cls in np.arange(len(np.unique(y)))]

# Create a Plotly 3D scatter plot
fig = go.Figure()
for cls, color in color_map.items():
    idx = [i for i, x in enumerate(y) if x == cls]
    fig.add_trace(go.Scatter3d(
        x=principal_components[idx, pcs[0]-1],
        y=principal_components[idx, pcs[1]-1],
        z=principal_components[idx, pcs[2]-1],
        mode='markers',
        marker=dict(color=color),
        name=f'{exp.class_ids[cls]}',
        hovertext=[f'Subject ID: {sid}' for sid in subject_ids[idx]],
        hoverinfo='text'
    ))

fig.update_layout(
    title=f'PC{pcs[0]} x PC{pcs[1]} x PC{pcs[2]}',
    scene=dict(
        xaxis_title=f'PC {pcs[0]}',
        yaxis_title=f'PC {pcs[1]}',
        zaxis_title=f'PC {pcs[2]}'
    ),
    legend_title='Class'
)
fig.show()

# %%
feat_of_interest = highlight_names[0]
feat_info = feat_of_interest.split('-')
modality = feat_info[0]
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

# %%
from ipywidgets import interact
from ants import image_read
import cv2
import SimpleITK as sitk

# %%
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
orientation = 'IAL'
mask_of_interest = get_seg_for_subject(subject_nums[0], int(roi), orientation=orientation)
scan_of_interest, scan_name = get_mri_for_subject(subject_nums[0], modality, orientation=orientation)

explore_3D_array_with_mask_contour(scan_of_interest, mask_of_interest, title = f'Subject {subject_nums[0]}, {pyrad_name}={round(subject_vals[0], 3)}, {modality} {roi_key[roi]}')

# %%
