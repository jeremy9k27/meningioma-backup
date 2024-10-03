# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import setup, lsdir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ants

setup()

data_dir = 'data/preprocessing/output/7b_COMPLETED_PREPROCESSED_SUBSET'
segs_dir = 'data/smooth_segs_9-4-24/'
segs_paths = [f for f in os.listdir(segs_dir) if f.startswith('Segmentation')]
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
biomarkers = ['MethylationSubgroup', 'Chr22q', 'Chr1p']
labels = pd.read_csv('data/labels/MeningiomaBiomarkerData.csv')

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

def read_cancerous_slice(scan_path):
    metadata = scan_path.split('/')
    subject = metadata[-4]
    seg = get_segs(subject, roi=22)
    cancerous_pixels_per_slice = np.sum(seg, axis=(1, 2))
    cancerous_slice = np.argmax(cancerous_pixels_per_slice)
    return ants.image_read(scan_path, reorient='IAL').numpy()[cancerous_slice, :, :]

def show_cancerous_slice(slice, label, fname):
    for spine in plt.gca().spines.values():
        spine.set_edgecolor(colours[int(label)])
        spine.set_linewidth(16)
    plt.imshow(slice, cmap='gray')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig(fname)

for subject in lsdir(data_dir):
    for session in lsdir(f'{data_dir}/{subject}'):
        for scan in lsdir(f'{data_dir}/{subject}/{session}'):
            for f in os.listdir(f'{data_dir}/{subject}/{session}/{scan}'):
                if f.endswith('.nii.gz'):
                    slice = read_cancerous_slice(f'{data_dir}/{subject}/{session}/{scan}/{f}')
                    for biomarker in biomarkers:
                        label = labels[labels['Subject Number'] == int(subject)][biomarker].values[0]
                        if np.isnan(label): label = 3
                        show_cancerous_slice(slice, label, fname=f'{data_dir}/{subject}/{session}/{scan}/Thumbnail_{biomarker}.png')
# %%
