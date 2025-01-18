# Package imports and directory setup:
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from segment_anything import sam_model_registry
from preprocessing.utils import setup, lsdir
from tqdm import tqdm
import numpy as np
import torch
import ants
import cv2

setup()

MRI_DIR = 'data/preprocessing/output/7b_COMPLETED_PREPROCESSED'
SEG_DIR = 'data/all_smooth_segs_12-12-24/'
SEGS_PATHS = [f for f in os.listdir(SEG_DIR) if f.startswith('Segmentation')]
OUTPUT_DIR = 'data/deeplearning_features/medsam'

# Load the encoder model:
encoder_model = sam_model_registry['vit_b'](checkpoint='data/encoder_models/medsam/medsam_vit_b.pth').image_encoder.to('cuda:0').eval()

# Helper functions:
def get_segs(subject, roi=22):
    """
    Given a subject ID number, returns volumetric segmentation mask for the specified region of interest (roi).

    Parameters:
    -----------
    subject (str or int): The subject ID number.
    roi (int): The region of interest (roi) to extract from the segmentation mask. Default is 22 (whole tumor mask).

    Returns:
    --------
    (np.ndarray or None): The volumetric segmentation mask for the specified region of interest (roi), 
    or None if the subject has no segmentation available or lacks the given roi. 
    Returned mask is binary, with 1s indicating the presence of the roi and 0s elsewhere.

    ROI Key:
    --------
    1: Enhancing tumor
    2: Other tumor
    3: Necrotic tumor
    4: Edema
    5: Susceptibility
    6: Resitricted diffusion
    7: Normal-appearing white matter (NAWM)
    13: Enhancing tumor + Necrotic tumor
    15: Enhancing tumor + Susceptibility
    16: Enhancing tumor + Resitricted diffusion
    156: Enhancing tumor + Susceptibility + Resitricted diffusion
    22: Whole tumor mask
    """
    all_seg_paths = [f for f in SEGS_PATHS if (f.startswith(f'Segmentation {subject}.nii') or f.startswith(f'Segmentation {subject} '))]
    if len(all_seg_paths) == 0: return None
    all_seg_arrays = []
    all_seg_labels = []
    for sp in all_seg_paths:
        seg_arr = ants.image_read(SEG_DIR + sp, reorient='IAL').numpy()
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
    
    if roi not in all_seg_labels: return None

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

def find_max_cancerous_slice(seg, axis=0):
    """
    Given a volumetric segmentation mask, returns the index of the slice along the provided axis with the maximum tumor content.

    Parameters:
    -----------
    seg (np.ndarray): The volumetric segmentation mask.
    axis (int): The axis along which to find the slice with the maximum tumor content. Default is 0 (axial plane if segmentation loaded in 'IAL' orientation with antspy).

    Returns:
    --------
    (int): The index of the slice with the maximum tumor content along the provided axis.
    """
    axes_to_sum_over = tuple(set(np.arange(seg.ndim)) - {axis})
    return np.argmax(np.sum(seg, axis=axes_to_sum_over))

def pad_im(im, desired_shape=(240, 240)):
    """
    Pads a np.ndarray to a desired shape by adding zeros around the image.
    """
    im_shape = im.shape
    height_diff = desired_shape[0] - im_shape[0]
    width_diff = desired_shape[1] - im_shape[1]
    pad_height = (height_diff // 2, height_diff // 2 + height_diff % 2)
    pad_width = (width_diff // 2, width_diff // 2 + width_diff % 2)
    return np.pad(im, (pad_height, pad_width), mode='constant', constant_values=0)

def preprocess_for_medsam(im):
    """
    Preprocesses an image for input into the MedSam encoder model. 
    
    From the MedSam paper:
    ----------------------
    "For MR... images, we clipped the intensity values to the range between the 0.5th and 99.5th percentiles before rescaling them to the range of [0, 255].
    Finally, to meet the model's input requirements, all images were resized to a uniform size of 1024 x 1024 x 3... 
    for 3D CT and MR images, each 2D slice was resized to 1024 x 1024, and the channel was repeated three times to maintain consistency...
    Bi-cubic interpolation was used for resizing images, while nearest-neighbor interpolation was applied for resizing masks to preserve their precise boundaries and avoid introducing unwanted artifacts."

    Warning:
    --------
    Didn't bother clipping intensity values to the 0.5th and 99.5th percentiles. 
    Also should maybe consider normalizing each image using the global min and max values across all images in the dataset.
    For now, haven't implemented either.
    """
    # pad image to 240x240:
    im_padded = pad_im(im)
    # resize image to 1024x1024:
    im_resized = cv2.resize(im_padded, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    # rescale image to [0, 255]:
    im_rescaled = (im_resized - im_resized.min()) / (im_resized.max() - im_resized.min()) * 255
    # copy image to make 3 channels:
    im_3c = np.repeat(im_rescaled[:, :, None], 3, axis=-1)
    # convert to tensor (add batch dimension and reorganize shape to be batch x channels x height x width):
    return torch.from_numpy(im_3c).unsqueeze(0).permute(0, 3, 1, 2).to('cuda:0')

# Main loop:
for subject in tqdm(lsdir(MRI_DIR), desc='Subjects', total=len(lsdir(MRI_DIR)), position=0, colour='green', dynamic_ncols=True):
    session = lsdir(f'{MRI_DIR}/{subject}')[0] # we take the first available session (alphabetically), others are ignored
    seg = get_segs(subject)
    if seg is None: continue # if subject has no segmentation available, we skip the subject
    for scan in tqdm(lsdir(f'{MRI_DIR}/{subject}/{session}'), desc='Scans', total=len(lsdir(f'{MRI_DIR}/{subject}/{session}')), position=1, colour='blue', dynamic_ncols=True, leave=False):
        scan_path = f'{MRI_DIR}/{subject}/{session}/{scan}/{session}_{scan}.nii.gz'
        # read image:
        mri = ants.image_read(scan_path, reorient='IAL').numpy()
        assert mri.shape == seg.shape
        # get slice w/most tumor content in the axial plane:
        max_cancerous_slice = find_max_cancerous_slice(seg)
        im = mri[max_cancerous_slice]
        # preprocess image for MedSam:
        im_tensor = preprocess_for_medsam(im)
        # encode image:
        with torch.no_grad():
            features = encoder_model(im_tensor).squeeze() # shape (256, 64, 64)
        # create output directory if it doesn't exist:
        cur_output_dir = f'{OUTPUT_DIR}/{subject}/{session}/{scan}'
        if not os.path.exists(cur_output_dir): os.makedirs(cur_output_dir)
        # save features:
        torch.save(features, f'{cur_output_dir}/{session}_{scan}.pt')