# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import ants
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

setup()

data_dir = 'data/preprocessing/output/7b_COMPLETED_PREPROCESSED_SUBSET'
segs_dir = 'data/smooth_segs_9-4-24/'
segs_paths = [f for f in os.listdir(segs_dir) if f.startswith('Segmentation')]

# %%
embedding_paths = []
for subject in os.listdir(data_dir):
    for session in os.listdir(f'{data_dir}/{subject}'):
        for scan in os.listdir(f'{data_dir}/{subject}/{session}'):
            st = scan.split('-')[-1]
            if st == 'AX_3D_T1_POST':
                for f in os.listdir(f'{data_dir}/{subject}/{session}/{scan}'):
                    if f.endswith('.h5'):
                        embedding_paths.append(f'{data_dir}/{subject}/{session}/{scan}/{f}')

print("Found", len(embedding_paths), "T1 POST embeddings")

# %%
def get_segs(subject, roi=22):
    all_seg_paths = [f for f in segs_paths if (f.startswith(f'Segmentation {subject}.nii') or f.startswith(f'Segmentation {subject} '))]
    all_seg_arrays = []
    all_seg_labels = []
    for sp in all_seg_paths:
        seg_arr = ants.image_read(segs_dir + sp, reorient='IAL').numpy()
        all_seg_arrays.append(seg_arr)
        all_seg_labels.extend([int(v) for v in np.unique(seg_arr) if v != 0])

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
                mask = np.logical_or(mask > 0, np.logical_and(seg_arr > 0, seg_arr != 7)) # we want to exclude the NAWM label = 7
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
        
        masks.append(mask)

    roi_idx = all_seg_labels.index(roi)
    mask_oi = masks[roi_idx]
    mask_oi = mask_oi > 0
    return mask_oi.astype(int)

# %%
def read_embedding(embedding_path):
    with h5py.File(embedding_path, 'r') as f:
        return f['embedding'][:]

def read_cancerous_slice(embedding_path):
    metadata = embedding_path.split('/')
    subject = metadata[-4]
    seg = get_segs(subject, roi=22)
    cancerous_pixels_per_slice = np.sum(seg, axis=(1, 2))
    cancerous_slice = np.argmax(cancerous_pixels_per_slice)
    with h5py.File(embedding_path, 'r') as f:
        return f['embedding'][cancerous_slice, :, :, :]

def pool_cancerous_slices(embedding_path, pool_type='avg'):
    assert pool_type in ['avg', 'max', 'min', 'median', 'sum', 'weighted_avg'], "Invalid pooling type, must be one of ['avg', 'max', 'min', 'median', 'sum', 'weighted_avg']"
    metadata = embedding_path.split('/')
    subject = metadata[-4]
    seg = get_segs(subject, roi=22)
    cancerous_pixels_per_slice = np.sum(seg, axis=(1, 2))
    cancerous_slices = np.where(cancerous_pixels_per_slice > 0)[0]
    with h5py.File(embedding_path, 'r') as f:
        embedding = f['embedding'][cancerous_slices, :, :, :]
        if pool_type == 'avg':
            return np.mean(embedding, axis=0)
        elif pool_type == 'max':
            return np.max(embedding, axis=0)
        elif pool_type == 'min':
            return np.min(embedding, axis=0)
        elif pool_type == 'median':
            return np.median(embedding, axis=0)
        elif pool_type == 'sum':
            return np.sum(embedding, axis=0)
        elif pool_type == 'weighted_avg':
            weights = cancerous_pixels_per_slice[cancerous_slices]
            return np.average(embedding, axis=0, weights=weights)

# %% 
# Populate embeddings
def get_embeddings(pooling, save=False):
    flattened_embeddings = []
    for ep in tqdm(embedding_paths, total=len(embedding_paths)):
        embedding = pool_cancerous_slices(ep, pooling) # read_cancerous_slice(ep)
        embedding_flat = embedding.flatten()
        flattened_embeddings.append(embedding_flat)
    flattened_embeddings = np.stack(flattened_embeddings)
    if save: np.save(f'data/tsne/t1post_{pooling}_pooled_embeddings.npy', flattened_embeddings)
    return flattened_embeddings

pooling_types = ['avg', 'max', 'min', 'median', 'sum', 'weighted_avg']
print(f"Step 1/4: Getting the embeddings for each of the following pooling types: {pooling_types}")

for pt in tqdm(pooling_types, total=len(pooling_types)):
    _ = get_embeddings(pt, save=True)

print("\nDone!\n")

# %%
# Populate labels
print("Step 2/4: Populating labels...")
labels_df = pd.read_csv('data/labels/MeningiomaBiomarkerData.csv')
labels = {'MethylationSubgroup': [], 'Chr1p': [], 'Chr22q': []}

for ep in tqdm(embedding_paths, total=len(embedding_paths)):
    subject = int(ep.split('/')[-4])
    for key in labels.keys():
        label = labels_df[labels_df['Subject Number'] == subject][key].values[0]
        labels[key].append(label)

print("\nDone!\n")

# %%
print("Step 3/4: PCA, UMAP Analyses...")
for pooling in tqdm(pooling_types, total=len(pooling_types)):
    flattened_embeddings = np.load(f'data/tsne/t1post_{pooling}_pooled_embeddings.npy')

    #### PCA ####
    pca = PCA(n_components=2, random_state=42)
    # pca_results = pca.fit_transform(StandardScaler().fit_transform(flattened_embeddings))
    pca_results = pca.fit_transform(flattened_embeddings)

    var = pca.explained_variance_ratio_

    pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
    for key in labels.keys():
        pca_df.loc[:, key] = np.stack(labels[key][:len(pca_results)])

    for key in labels.keys():
        g = sns.relplot(data=pca_df, x='PC1', y='PC2', hue=key, palette='tab10')
        g.figure.suptitle(f'PCA {pooling} Pooled Embeddings: {key}\nVar Exp: PC1={round(var[0], 2)}, PC2={round(var[1], 2)}', y=1.02)
        g.savefig(f'data/tsne/PCA_{pooling}_pooled_embeddings_{key}.png')

    #### UMAP ####
    umap_results = umap.UMAP(n_components=2, random_state=42).fit_transform(flattened_embeddings)

    umap_df = pd.DataFrame(umap_results, columns=['x', 'y'])
    for key in labels.keys():
        umap_df.loc[:, key] = np.stack(labels[key][:len(umap_results)])

    for key in labels.keys():
        g = sns.relplot(data=umap_df, x='x', y='y', hue=key, palette='tab10')
        g.figure.suptitle(f'UMAP {pooling} Pooled Embeddings: {key}', y=1.02)
        g.savefig(f'data/tsne/UMAP_{pooling}_pooled_embeddings_{key}.png')

print("\nDone!\n")

# %%
#### TSNE ANALYSIS ####
def tsne_analysis(X, y, perplexities=[2, 5, 10, 20, 30], n_iters=[250, 1000, 2500, 5000, 7500], random_state=42):
    results = pd.DataFrame()
    for perp in tqdm(perplexities, total=len(perplexities), colour='green', position=0, leave=True):
        for n_it in tqdm(n_iters, total=len(n_iters), colour='blue', position=1, leave=False):
            tsne = TSNE(n_components=2, perplexity=perp, max_iter=n_it, random_state=random_state)
            tsne_results = tsne.fit_transform(X)
            df = pd.DataFrame(tsne_results, columns=['x', 'y'])
            for key in y.keys():
                df.loc[:, key] = np.stack(y[key])
            df.loc[:, 'perplexity'] = perp
            df.loc[:, 'n_iter'] = n_it
            results = pd.concat([results, df], axis=0)
    return results

print("Step 4/4: TSNE Analysis...")
for pooling in tqdm(pooling_types, total=len(pooling_types), desc='Pooling Types', colour='red'):
    flattened_embeddings = np.load(f'data/tsne/t1post_{pooling}_pooled_embeddings.npy')
    results_df = tsne_analysis(flattened_embeddings, labels)
    results_df.to_csv(f'data/tsne/t1post_{pooling}_pooled_TSNE_ANALYSIS_embeddings.csv', index=False)
    # Plot and save results
    results_df = pd.read_csv(f'data/tsne/t1post_{pooling}_pooled_TSNE_ANALYSIS_embeddings.csv')
    for key in labels.keys():
        g = sns.relplot(data=results_df, x='x', y='y', hue=key, palette='tab10', col='perplexity', row='n_iter', facet_kws={'sharex': False, 'sharey': False})
        g.figure.set_size_inches(16, 16)
        g.figure.suptitle(f'TSNE {pooling} Pooled Embeddings: {key}', y=1.02)
        g.savefig(f'data/tsne/TSNE_ANALYSIS_{pooling}_pooled_embeddings_{key}.png')

print("\nDone with script!\n")

#### 3D TSNE ####
# tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
# tsne_results = tsne.fit_transform(flattened_embeddings)

# tsne_df = pd.DataFrame(tsne_results, columns=['x', 'y', 'z'])
# for key in labels.keys():
#     tsne_df.loc[:, key] = np.stack(labels[key][:len(tsne_results)])

# # Set Seaborn style
# sns.set_style("whitegrid")

# for key in labels.keys():
#     fig = px.scatter_3d(tsne_df, x='x', y='y', z='z', color=key, opacity=0.7)
#     fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#     fig.show()

# %%
# tsne = TSNE(n_components=2, perplexity=20, max_iter=1000, random_state=1)
# tsne_results = tsne.fit_transform(flattened_embeddings)

# tsne_df = pd.DataFrame(tsne_results, columns=['x', 'y'])
# for key in labels.keys():
#     tsne_df.loc[:, key] = np.stack(labels[key][:len(tsne_results)])

# for key in labels.keys():
#     sns.relplot(data=tsne_df, x='x', y='y', hue=key, palette='tab10')

# %%
def plot_embeddings_w_thumbs(biomarker, pooling, dim_red='pca', precomputed_space=None, legend_loc='lower right'):
    
    # Build up list of thumbnail paths, labels to plot
    def get_thumbs_and_labs(scan_types_of_interest=['AX_3D_T1_POST']):
        '''Loops thru data_dir and returns thumbnail paths, labels for the specified biomarker and scan types of interest'''
        paths = []
        labels = []

        # Read in labels for each subject
        labels_df = pd.read_csv('data/labels/MeningiomaBiomarkerData.csv')

        # Loop thru all subjects and retrieve thumbnails, labels for the specified biomarker, scan types
        for subject in os.listdir(data_dir):
            for session in os.listdir(f'{data_dir}/{subject}'):
                for scan in os.listdir(f'{data_dir}/{subject}/{session}'):
                    st = scan.split('-')[-1]
                    if st in scan_types_of_interest:
                        for f in os.listdir(f'{data_dir}/{subject}/{session}/{scan}'):
                            if f.endswith(f'{biomarker}.png'):
                                # Append thumbnail
                                paths.append(f'{data_dir}/{subject}/{session}/{scan}/{f}')
                                label = labels_df[labels_df['Subject Number'] == int(subject)][biomarker].values[0]
                                labels.append(label)

        return paths, labels

    thumbnail_paths, labels = get_thumbs_and_labs()

    # Load embeddings for the specified pooling type
    flattened_embeddings = np.load(f'data/tsne/t1post_{pooling}_pooled_embeddings.npy')

    # We only want to plot the embeddings for which we have labels
    idxs_w_labels = np.where(~np.isnan(labels))[0]
    thumbnail_paths = [thumbnail_paths[i] for i in idxs_w_labels]

    # Being plot
    fig, ax = plt.subplots(figsize=(20, 20))

    if precomputed_space is not None:
        x, y = precomputed_space
        subtitle = ''
    else:
        if dim_red == 'pca':
            pca = PCA(n_components=2, random_state=42)
            pca_results = pca.fit_transform(flattened_embeddings)
            x = pca_results[:, 0]
            y = pca_results[:, 1]
            var = pca.explained_variance_ratio_
            subtitle = '\nVar Exp: PC1={:.1f}%, PC2={:.1f}%'.format(var[0]*100, var[1]*100)
            ax.set_xlabel('PC1', fontsize=24)
            ax.set_ylabel('PC2', fontsize=24)
            
        elif dim_red == 'umap':
            umap_results = umap.UMAP(n_components=2, random_state=42).fit_transform(flattened_embeddings)
            x = umap_results[:, 0]
            y = umap_results[:, 1]

            subtitle = ''
            ax.set_xlabel('UMAP1', fontsize=24)
            ax.set_ylabel('UMAP2', fontsize=24)

        # We only want to plot the embeddings for which we have labels
        x = x[idxs_w_labels]
        y = y[idxs_w_labels]

    # Begin plot
    ax.scatter(x, y)

    # Plot thumbnails
    def get_image(path):
        '''Returns an OffsetImage object for the given path, cropped and zoomed to specifically fit the meningioma thumbnails made in preprocessing script 8c'''
        img = Image.open(path)
        img = img.crop((160, 45, 495, 438))
        return OffsetImage(img, zoom=0.20)

    for x0, y0, path in zip(x, y, thumbnail_paths):
        ab = AnnotationBbox(get_image(path), (x0, y0), frameon=False)
        ax.add_artist(ab)

    # Add legend
    if biomarker != 'MethylationSubgroup':
        patch0 = mpatches.Patch(color='tab:blue', label='Intact')
        patch1 = mpatches.Patch(color='tab:orange', label='Lost')
        ax.legend(handles=[patch0, patch1], loc=legend_loc, fontsize=24)
    else:
        patch0 = mpatches.Patch(color='tab:blue', label='Merlin Intact')
        patch1 = mpatches.Patch(color='tab:orange', label='Immune Enriched')
        patch2 = mpatches.Patch(color='tab:green', label='Hypermetabolic')
        ax.legend(handles=[patch0, patch1, patch2], loc=legend_loc, fontsize=24)

    # Set title

    ax.set_title(f'Algo: {dim_red.upper()}, Pooling: {pooling.capitalize()}, Labels: {biomarker}{subtitle}', fontsize=36)
    plt.show()

    return x, y

# %%

pca_x, pca_y = plot_embeddings_w_thumbs('Chr22q', 'weighted_avg', 'pca')
# pca_x, pca_y = plot_embeddings_w_thumbs('Chr22q', 'weighted_avg', 'pca', precomputed_space = (pca_x, pca_y))
# %%
umap_x, umap_y = plot_embeddings_w_thumbs('Chr22q', 'weighted_avg', 'umap')

# %%
umap_x, umap_y = plot_embeddings_w_thumbs('MethylationSubgroup', 'weighted_avg', 'umap')

# %%
pooling_types = ['avg', 'max', 'min', 'median', 'sum', 'weighted_avg']
for pt in pooling_types:
    _, _ = plot_embeddings_w_thumbs('Chr22q', pt, 'umap')
# %%
