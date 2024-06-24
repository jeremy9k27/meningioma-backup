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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
import plotly.graph_objects as go
from ipywidgets import interact, IntSlider
from ants import image_read
import cv2
from LTOExperiment import LTOExperiment
from LOOExperiment import LOOExperiment
import seaborn as sns

setup()

# directory handling
data_dir = 'results/PyRad_Collage_PCA/'
task = 'MethylationSubgroup'
output_dir = f'{data_dir}/{task}'
if not os.path.exists(output_dir): os.makedirs(output_dir)

# Get data prepared
X = prep_data_for_pca(features_file='data/combined_feats/radiomics8-smoothed_collage-ws-5-bs-32_features.csv', outcome=task, scaler_obj=StandardScaler())
subject_ids = X['Subject Number']
y = X[task]
plot_data_split(y, task, output_file=f'{output_dir}/data_split.png')
X = X.drop(columns=['Subject Number', task])
feat_names = X.columns

# %%
pca = PCA()
principal_components = pca.fit_transform(X)
# pca_scores_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
# pca_scores_df['Subject Number'] = subject_ids

# if not os.path.exists(f"{output_dir}/pca_scores.csv"): 
#     pca_scores_df.to_csv(f"{output_dir}/pca_scores.csv", index=False)

is_collage = np.array(['ang' in col_name for col_name in X.columns])

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
# feature_type = [feat.split('-')[-1].split('_')[0] for feat in X.columns]
# roi_type = [feat.split('-')[1] for feat in X.columns]
# modality_type = [feat.split('-')[0] for feat in X.columns]
# pyrad_name_type = [feat.split('_')[-1] for feat in X.columns]

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
loadings = pca.components_.T

def plot_loadings(n_row=1, n_col=5, PCs=None, by_sign=True, output_dir=None):
    if PCs is None:
        PCs = np.arange(1, (n_row*n_col) + 1).astype(int)
    
    fig, axes = plt.subplots(n_row, n_col, figsize=(10, 10))
    fig.suptitle(f'{task} PCA Loadings', y = 1.05)
    ax = axes.flatten()

    for i in range(len(ax)):
        var_exp = round(variance_df['Variance Explained'][PCs[i]-1], 2)
        cum_var_exp = round(variance_df['Cumulative Variance'][PCs[i]-1], 2)
        sorted_indices = np.argsort((loadings[:, PCs[i]-1]))[::-1]
        if by_sign: 
            colors = ['tab:blue' if x > 0 else 'tab:orange' for x in loadings[sorted_indices, PCs[i]-1]]
        else:
            colors = ['tab:blue' if is_c else 'tab:orange' for is_c in is_collage]
        ax[i].scatter(range(len(feat_names)), (loadings[sorted_indices, PCs[i]-1]), c=colors, alpha=0.1, s=5)
        ax[i].set_title(f'PC{PCs[i]} (Var: {var_exp}, Cum: {cum_var_exp})')
        if i >= (n_row*n_col) - n_col: ax[i].set_xlabel('Features')
        if i % n_row == 0: ax[i].set_ylabel('Loading Value')

    plt.tight_layout()
    if output_dir is not None:
        str_pcs = "-".join([str(pc) for pc in PCs])
        plt.savefig(f'{output_dir}/loadings_pcs_{str_pcs}.png', bbox_inches='tight')
    else:
        plt.show()
    plt.close()

plot_loadings(1, 2, by_sign=False, output_dir=output_dir)
plot_loadings(2, 3, by_sign=False, output_dir=output_dir)

# %%
corr_mat = X.corr()

# %%
colors = ['tab:blue' if is_c else 'tab:orange' for is_c in is_collage]
g = sns.clustermap(
    corr_mat, 
    row_cluster=False, col_cluster=False,
    linewidths=0, xticklabels=False, yticklabels=False,
    row_colors=colors, col_colors=colors, cmap='coolwarm', figsize=(12, 12)
)

# %% rank 1 reconstruction
loadings = pca.components_.T
rank1_recon = pd.DataFrame(np.dot(principal_components[:, :6], loadings[:, :6].T), columns=feat_names)

# %%
rank1_recon_corr = rank1_recon.corr()

# %%
colors = ['tab:blue' if is_c else 'tab:orange' for is_c in is_collage]
g_rank1 = sns.clustermap(
    rank1_recon_corr, 
    row_cluster=False, col_cluster=False,
    linewidths=0, xticklabels=False, yticklabels=False,
    row_colors=colors, col_colors=colors, cmap='coolwarm', figsize=(12, 12)
)
g_rank1.savefig('results/lowrank_corr_mat.png')

# %%
# Map classes to colors
if exp.n_classes > 2:
    color_map = {0: 'blue', 1: 'green', 2: 'red'}
else:
    color_map = {0: 'blue', 1: 'red'}
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
plt.tight_layout()
plt.savefig(f'{output_dir}/3d_pca_triplot.png', bbox_inches='tight')
plt.close()
# plt.show()

# %%

# Map classes to colors
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
    sliders = {f'Slice {i}': IntSlider(min=0, max=arr_list[i].shape[0]-1, value=max_slice_list[i], description=f'Slice {i}') for i in range(num_images)}

    def update_slices(**slices):
        if len(titles) == 3:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        if num_images == 1:
            axes = [axes]

        fig.suptitle(suptitle)

        for i, (arr, mask, ax) in enumerate(zip(arr_list, mask_list, axes.flatten())):
            slice_idx = slices[f'Slice {i}']
            arr_rgb = cv2.cvtColor(arr[slice_idx, :, :], cv2.COLOR_GRAY2RGB)
            contours, _ = cv2.findContours(mask[slice_idx, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0, 255, 0), thickness)
            
            ax.imshow(arr_with_contours)
            ax.axis('off')
            ax.set_title(titles[i] if titles else f'Image {i}')

        plt.show()

    interact(update_slices, **sliders)

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
