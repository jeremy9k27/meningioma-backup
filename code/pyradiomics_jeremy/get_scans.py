import pandas as pd
import numpy as np
# logging helpers
from tqdm import tqdm
from zlib import adler32
from radiomics import featureextractor
import SimpleITK as sitk
from torch.utils.data import Dataset
import os
import nibabel as nib
import torch.nn as nn
import torch
from torchvision import models
import pickle


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
    #seg[seg!=3] = 0
    seg[seg>0] = 1
    axes_to_sum_over = tuple(set(np.arange(seg.ndim)) - {axis})
    max_index = np.argmax(np.sum(seg, axis=axes_to_sum_over))
    max_amount = np.sum(seg[ :, :, max_index])
    return max_index, max_amount


class Scan:
    def __init__(self, scan_path, seg_path):
        self.scan_path = scan_path
        self.seg_path = seg_path 
        self.patient_num = 0 
        self.scan_type = 0
        self.pyradiomics_features = {}
        self.efficientnet_rep = []

    def get_scan_np(self):
        return nib.load(self.scan_path).get_fdata()

    def get_seg_np_filtered(self):
        seg_np = nib.load(self.seg_path).get_fdata()
        return (seg_np > 0).astype(np.uint8)

    def set_efficientnet_rep(self, efficientnet):
        max_cancer_index, max_amount = find_max_cancerous_slice(self.get_seg_np_filtered(), axis=2)
        max_cancer_slice = np.flip(np.rot90(self.get_scan_np()[ :, :, max_cancer_index],k=3), axis = 1)

        slice_tensor = torch.tensor(max_cancer_slice, dtype=torch.float32)
        slice_tensor = slice_tensor / slice_tensor.max()
        slice_tensor = slice_tensor.unsqueeze(0)
        slice_tensor = slice_tensor.repeat(3, 1, 1)  # shape becomes (3, H, W)
        slice_tensor = slice_tensor.unsqueeze(0)
        
        features = efficientnet(slice_tensor)     
        self.efficientnet_rep = features 

        return max_cancer_slice

    def set_pyradiomics_features(self, extractor):
        scan_sitk = sitk.GetImageFromArray(self.get_scan_np())
        mask_sitk = sitk.GetImageFromArray(self.get_seg_np_filtered())
        features = extractor.execute(scan_sitk, mask_sitk)

        feature_names = [
                        "original_firstorder_Energy", 
                        "original_firstorder_Entropy", 
                        "original_firstorder_Mean", 
                        "original_firstorder_Median", 
                        "original_firstorder_StandardDeviation", 
                        "original_firstorder_Kurtosis", 
                        "original_firstorder_Uniformity", 
                        "original_shape_VoxelVolume", 
                        "original_shape_SurfaceArea", 
                        "original_shape_SphericalDisproportion"
                        ]

        for feature in feature_names:
            self.pyradiomics_features[feature] = features.get(feature, "Not Found")



def main():
    OUTPUT_FILE = 'features.csv'
    extractor = featureextractor.RadiomicsFeatureExtractor()

    features_row = pd.concat([pd.DataFrame({'Subject Number': [s], 'Modality': [modality], 'Segmentation Label': all_seg_labels[i]}), pd.Series(result_trimmed).to_frame().T], axis=1)
    features_row.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)

    efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)  # Keyword
    efficientnet.classifier = nn.Identity()
    scans = []

    for file_name in os.listdir("BraTS-MEN-Train"):
        file_path = os.path.join("BraTS-MEN-Train", file_name) 
        print(file_path)

        for scan_type in ["-t1c.nii.gz", "-t1n.nii.gz", "-t2f.nii.gz", "-t2w.nii.gz"]:
            segmented_path = file_path + "/" + file_path[16:] + "-seg.nii.gz"
            scan_path = file_path + "/" + file_path[16:] + scan_type

            scan = Scan(scan_path, segmented_path)
            #scan.set_pyradiomics_features(extractor)
            #scan.set_efficientnet_rep(efficientnet)
            scan.scan_type = scan_type[1:4]
            scan.patient_num = int(file_path[-9:-4])
            scans.append(scan)

    with open('scans.pkl', 'wb') as f:
        pickle.dump(scans, f)
        
if __name__ == "__main__":
    main()