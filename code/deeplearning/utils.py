import numpy as np
from ants import image_read
from preprocessing.utils import lsdir

def get_seg_roi_key():
    """Returns a dict mapping int keys to corresponding segmentation rois."""
    return {
        1: 'Enhancing tumor',
        2: 'Other tumor',
        3: 'Necrotic tumor',
        4: 'Edema',
        5: 'Susceptibility',
        6: 'Resitricted diffusion',
        7: 'Normal-appearing white matter (NAWM)',
        13: 'Enhancing tumor + Necrotic tumor',
        15: 'Enhancing tumor + Susceptibility',
        16: 'Enhancing tumor + Resitricted diffusion',
        156: 'Enhancing tumor + Susceptibility + Resitricted diffusion',
        22: 'Whole tumor mask',
    }

def get_segs(subject, seg_dir, seg_paths, rois=[1, 3, 4, 5, 6]):
    """
    Given a subject ID number, returns volumetric segmentation mask for the specified region of interest (roi).

    Parameters:
    -----------
    subject (str or int): The subject ID number.
    seg_dir (str): The directory containing the segmentation masks.
    seg_paths (list): List of segmentation mask file names.
    roi (int): The region of interest (roi) to extract from the segmentation mask. Default is 22 (whole tumor mask).

    Returns:
    --------
    (np.ndarray or None): The volumetric segmentation mask for the specified region of interest (roi), 
    or None if the subject has no segmentation available or lacks the given roi. 
    Returned mask is binary, with 1s indicating the presence of the roi and 0s elsewhere.

    ROI Key:
    --------
    1: Enhancing tumor*
    2: Other tumor
    3: Necrotic tumor*
    4: Edema*
    5: Susceptibility*
    6: Resitricted diffusion*
    7: Normal-appearing white matter (NAWM)
    13: Enhancing tumor + Necrotic tumor
    15: Enhancing tumor + Susceptibility
    16: Enhancing tumor + Resitricted diffusion
    156: Enhancing tumor + Susceptibility + Resitricted diffusion
    22: Whole tumor mask
    """
    if not seg_dir.endswith('/'): seg_dir += '/'
    all_seg_paths = [f for f in seg_paths if (f.startswith(f'Segmentation {subject}.nii') or f.startswith(f'Segmentation {subject} '))]
    if len(all_seg_paths) == 0: return None
    all_seg_arrays = []
    all_seg_labels = []
    for sp in all_seg_paths:
        seg_arr = image_read(seg_dir + sp, reorient='IAL').numpy()
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

    all_seg_labels = sorted(list(set(all_seg_labels)))
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

    # grab those masks that correspond to the requested rois
    final_masks = {}
    for roi in rois:
        if roi not in all_seg_labels:
            final_masks[roi] = np.zeros_like(masks[0])
        else:
            roi_idx = all_seg_labels.index(roi)
            mask_oi = masks[roi_idx]
            mask_oi = mask_oi > 0
            final_masks[roi] = mask_oi.astype(int)

    return final_masks

def get_mris(subject_mri_dir, pulse_sequences):
    session = subject_mri_dir.split('/')[-1]
    mris = {}
    scans = lsdir(subject_mri_dir)
    for ps in pulse_sequences:
        for scan in scans:
            if scan.lower().endswith(ps.lower()):
                mris[ps] = image_read(f'{subject_mri_dir}/{scan}/{session}_{scan}.nii.gz', reorient='IAL').numpy()
                break
    return mris

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