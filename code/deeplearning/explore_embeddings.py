# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import ants
import seaborn as sns

setup()

data_dir = 'data/preprocessing/output/7b_COMPLETED_PREPROCESSED_SUBSET'
segs_dir = 'data/smooth_segs_9-4-24/'
segs_paths = [f for f in os.listdir(segs_dir) if f.startswith('Segmentation')]

embedding_paths = []
for subject in os.listdir(data_dir):
    for session in os.listdir(f'{data_dir}/{subject}'):
        for scan in os.listdir(f'{data_dir}/{subject}/{session}'):
            for f in os.listdir(f'{data_dir}/{subject}/{session}/{scan}'):
                if f.endswith('.h5'):
                    embedding_paths.append(f'{data_dir}/{subject}/{session}/{scan}/{f}')

print("Found", len(embedding_paths), "embeddings")

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

# %%
labels_df = pd.read_csv('data/labels/MeningiomaBiomarkerData.csv')
flattened_embeddings = []
labels = {'MethylationSubgroup': [], 'Chr1p': [], 'Chr22q': []}
for ep in tqdm(embedding_paths, total=len(embedding_paths)):
    subject = int(ep.split('/')[-4])
    embedding = read_cancerous_slice(ep)
    embedding_flat = embedding.flatten()
    flattened_embeddings.append(embedding_flat)
    for key in labels.keys():
        label = labels_df[labels_df['Subject Number'] == subject][key].values[0]
        labels[key].append(label)

flattened_embeddings = np.stack(flattened_embeddings)

# %%
def tsne_analysis(X, y, perplexities=[2, 5, 10, 20, 30], n_iters=[250, 1000, 2500, 5000, 7500], random_state=42):
    results = pd.DataFrame()
    for perp in tqdm(perplexities, total=len(perplexities), colour='green', position=0, leave=True):
        for n_it in tqdm(n_iters, total=len(n_iters), colour='blue', position=1, leave=False):
            tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_it, random_state=random_state)
            tsne_results = tsne.fit_transform(X)
            df = pd.DataFrame(tsne_results, columns=['x', 'y'])
            for key in y.keys():
                df.loc[:, key] = np.stack(y[key])
            df.loc[:, 'perplexity'] = perp
            df.loc[:, 'n_iter'] = n_it
            results = pd.concat([results, df], axis=0)
    return results

# tsne = TSNE(n_components=2, perplexity=2, n_iter=5000, random_state=42)
# tsne_results = tsne.fit_transform(flattened_embeddings)

# %%
results_df = tsne_analysis(flattened_embeddings, labels)
results_df.to_csv('data/tsne/cancerous_slice_embeddings.csv', index=False)

# %%
# Plot and save results
for key in labels.keys():
    g = sns.relplot(data=results_df, x='x', y='y', hue=key, palette='tab10', col='perplexity', row='n_iter')
    g.figure.set_size_inches(16, 16)
    g.figure.suptitle(f'Cancerous Slice Embeddings: {key}', y=1.02)
    g.savefig(f'data/tsne/cancerous_slice_embeddings_{key}.png')

# %%
