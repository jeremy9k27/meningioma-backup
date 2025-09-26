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
import matplotlib.pyplot as plt



def main():
    OUTPUT_FILE = 'code/pyradiomics_jeremy/efficientnet_results.csv'
    

    efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)  # Keyword
    efficientnet.classifier = nn.Identity()
    

    for file_name in tqdm(os.listdir("BraTS-MEN-Train")):
        file_path = os.path.join("BraTS-MEN-Train", file_name) 
        print(file_path)

        for scan_type in ["-t1c.nii.gz", "-t1n.nii.gz", "-t2f.nii.gz", "-t2w.nii.gz"]:
            seg_path = file_path + "/" + file_path[16:] + "-seg.nii.gz"
            scan_path = file_path + "/" + file_path[16:] + scan_type

            modality = scan_type[1:4]
            patient_num = int(file_path[-9:-4])
            
            
            scan = nib.load(scan_path).get_fdata()
            latent_vectors = []

            for i in range(scan.shape[2]):
                slice_2d = scan[:, :, i]
                if np.all(slice_2d == 0):
                    continue     
                #    dont rescale, just pass in the zeros       
                slice_2d = np.flip(np.rot90(slice_2d,k=3), axis = 1)
                

                slice_tensor = torch.tensor(slice_2d, dtype=torch.float32)
                slice_tensor = slice_tensor / slice_tensor.max()
                slice_tensor = slice_tensor.unsqueeze(0)
                slice_tensor = slice_tensor.repeat(3, 1, 1)  # shape becomes (3, H, W)
                slice_tensor = slice_tensor.unsqueeze(0)

                features = efficientnet(slice_tensor)   
                latent_vectors.append(features)

            latent_tensor = torch.stack(latent_vectors)
            avg_pooled = torch.mean(latent_tensor, dim=0)
            max_pooled = torch.max(latent_tensor, dim=0).values


            avg_vector = avg_pooled.squeeze().detach().cpu().numpy()
            max_vector = max_pooled.squeeze().detach().cpu().numpy()
            avg_series = pd.Series(avg_vector, index=[f'avg_feature_{i}' for i in range(len(avg_vector))])
            max_series = pd.Series(max_vector, index=[f'max_feature_{i}' for i in range(len(max_vector))])

            # Combine metadata with both feature sets
            features_row = pd.concat([pd.DataFrame({'Patient Number': [patient_num], 'Modality': [modality]}), avg_series.to_frame().T,max_series.to_frame().T], axis=1)
            
            if not os.path.exists(OUTPUT_FILE):
                features_row.to_csv(OUTPUT_FILE, index=False)
            else:
                features_row.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)
            
        
if __name__ == "__main__":
    main()