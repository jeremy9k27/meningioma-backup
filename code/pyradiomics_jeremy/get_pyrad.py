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
import multiprocessing
import concurrent.futures
import csv
import time


def safe_execute(pipe, scan_sitk, seg_sitk):
    try:
        # Re-initialize the extractor in the child process
        from radiomics import featureextractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()

        result = extractor.execute(scan_sitk, seg_sitk)
        pipe.send(result)
    except Exception as e:
        pipe.send(e)


def get_segmentation(seg_path, seg_type):
    seg = nib.load(seg_path).get_fdata()
    seg[~np.isin(seg, seg_type)] = 0
    #print(seg_type, np.unique(seg), np.count_nonzero(seg))
    seg[seg>0] = 1
    
    return seg



def main():

    TIMEOUT_SECONDS = 30
    OUTPUT_FILE = 'code/pyradiomics_jeremy/features.csv'
    existing_row_count = get_existing_row_count(OUTPUT_FILE)
    seg_types = [(1,), (2,), (3,), (1,3), (2,3), (1,2,3)]


    #set up pyradiomics feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()

    #set up efficientnet
    efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)  # Keyword
    efficientnet.classifier = nn.Identity()
    
    malformed_patient_id = 0

    # iter through patients
    for file_name in tqdm(os.listdir("BraTS-MEN-Train")):
        file_path = os.path.join("BraTS-MEN-Train", file_name) 

        #iter through scan type
        for scan_type in ["-t1c.nii.gz", "-t1n.nii.gz", "-t2f.nii.gz", "-t2w.nii.gz"]:

            seg_path = file_path + "/" + file_path[16:] + "-seg.nii.gz"
            scan_path = file_path + "/" + file_path[16:] + scan_type

            modality = scan_type[1:4]
            patient_num = int(file_path[-9:-4])

            if patient_num == malformed_patient_id:
                    continue   
            
            scan_sitk = sitk.GetImageFromArray(nib.load(scan_path).get_fdata())
            
            # iter through segmentation types
            for seg_type in seg_types:
                
                seg = get_segmentation(seg_path, seg_type)
                if np.all(seg == 0):
                    continue
    
                seg_sitk = sitk.GetImageFromArray(seg)

                # feature extraction. timeout if it takes longer than 30 seconds. a "clean" iteration takes ~3 seconds, so 30
                # is pretty conservative
                parent_conn, child_conn = multiprocessing.Pipe()
                p = multiprocessing.Process(target=safe_execute, args=(child_conn, scan_sitk, seg_sitk))
                p.start()
                p.join(TIMEOUT_SECONDS)

                if p.is_alive():
                    p.terminate()
                    p.join()
                    print(f"Timeout: extractor.execute took too long for {scan_path}, {seg_type}")
                    continue

                result_or_exception = parent_conn.recv()
                if isinstance(result_or_exception, Exception):
                    print(f"Failed to load scan image for {scan_path}, {seg_type}: {result_or_exception}")
                    malformed_patient_id = patient_num
                    continue

                result = result_or_exception




                result_trimmed = result.copy()
                for k in result.keys():
                    if not isinstance(result[k], np.ndarray):
                        del result_trimmed[k]
                    elif result[k].size > 1:        
                        del result_trimmed[k]

                features_row = pd.concat([pd.DataFrame({'Patient Number': [patient_num], 'Modality': [modality], 'Segmentation Label': [seg_type]}), pd.Series(result_trimmed).to_frame().T], axis=1)
                if not os.path.exists(OUTPUT_FILE):
                    features_row.to_csv(OUTPUT_FILE, index=False)
                else:
                    features_row.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)
        
        
if __name__ == "__main__":
    main()