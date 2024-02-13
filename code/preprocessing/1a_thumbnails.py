# File: 1c_thumbnails.py
# Date: 12/30/2023
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Generates .png thumbnails for DICOMs

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports
# 1. Helper functions
# 2. Set up filepaths
# 3. Generate thumbnails

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script generates .png thumbnails for the middle slice of a scan for all
# DICOMs in the ask_virginia dirs of the data directory. This way, Virginia can
# inspect the DICOMs by hand quickly to easily rename the files appropriately.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/NURIPS_downloads/Meningiomas_handchecked/*/*_Brainlab/ask_virginia/*/resources/DICOM/*.dcm
#
# This script generates the following file(s) as outputs:
#   * data/preprocessing/output/thumbnails/*/*.png
#   * data/preprocessing/output/thumbnails/*/*_README.txt

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
import os
import pydicom
from PIL import Image
import numpy as np

#%%-------------------------#
#### 1. HELPER FUNCTIONS ####
#---------------------------#
def make_thumbnail(directory, output_file):
    """
    Make a thumbnail .png of the middle slice from a directory of DICOMs
    """
    dicoms = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.dcm')]
    dicoms.sort(key=lambda x: int(x.InstanceNumber))
    middle_idx = len(dicoms) // 2
    image_2d = dicoms[middle_idx].pixel_array.astype(float)
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    image_2d_scaled = np.uint8(image_2d_scaled)
    image = Image.fromarray(image_2d_scaled)
    image.save(output_file)

def lsdir(path):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

#%%-------------------------#
#### 2. SET UP FILEPATHS ####
#---------------------------#
if not os.getcwd().endswith('Meningioma'): os.chdir('../..')
if not os.getcwd().endswith('Meningioma'): 
    raise Exception('Please run this script from the Menigioma directory')

data_dir = 'data/preprocessing/NURIPS_downloads/Meningiomas_handchecked'
alt_dir = 'data/preprocessing/NURIPS_downloads/Meningiomas'
dcm = 'resources/DICOM'
output_dir = 'data/preprocessing/output/1_scan_type_cleanup/1c_thumbnails'
if not os.path.exists(output_dir): os.makedirs(output_dir)

#%%----------------------------#
#### 3. GENERATE THUMBNAILS ####
#------------------------------#
for subject in lsdir(data_dir):
    for session in lsdir(f'{data_dir}/{subject}'):
        save_to = f'{output_dir}/{session}'
        if not os.path.exists(save_to): os.makedirs(save_to)
        readme = f'{save_to}/{subject}_README.txt'
        with open(readme, 'w') as f:
            f.write(f'Patient: {subject}\n')
            f.write(f'Session: {session}\n\n')
            f.write('=' * 80 + '\n\n')
            f.write('Scans that need renaming or removing (see associated thumbnails):\n\n')
            for scan in lsdir(f'{data_dir}/{subject}/{session}/ask_virginia'):
                make_thumbnail(f'{data_dir}/{subject}/{session}/ask_virginia/{scan}/{dcm}', f'{save_to}/{scan}.png')
                f.write(f'{scan} --> \n')
            f.write('\n' + '=' * 80 + '\n\n')
            f.write('Metadata that may provide helpful context\n(feel free to ignore if thumbnails alone are sufficient):\n\n')
            f.write('-' * 80 + '\n')
            f.write('Scans that are already ready for preprocessing:\n\n')
            for scan in lsdir(f'{data_dir}/{subject}/{session}/ready_for_preprocessing'):
                f.write(f'{scan}\n')
            f.write('-' * 80 + '\n')
            f.write('Discarded scans:\n\n')
            for scan in lsdir(f'{alt_dir}/{subject}/{session}/scans'):
                f.write(f'{scan}\n')
            f.write('-' * 80 + '\n')

# %%
