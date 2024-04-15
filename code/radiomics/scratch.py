# %%
import collageradiomics
import SimpleITK as sitk
import numpy as np
import os
import sys
import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup

setup()

# %%
im1 = sitk.ReadImage('data/preprocessing/output/7_COMPLETED_PREPROCESSED/6/6_Brainlab/12-AX_3D_T1_POST/6_Brainlab_12-AX_3D_T1_POST.nii.gz')
mask1 = sitk.ReadImage('data/segmentations/Segmentation 6.nii')
# %%
textures = collageradiomics.Collage(sitk.GetArrayFromImage(im1), sitk.GetArrayFromImage(mask1)).execute()

# %%
from Experiment import Experiment
import time
from datetime import datetime
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup

setup()

exp = Experiment(
    prediction_task='MethylationSubgroup', 
    test_size=22, 
    seed=1, 
    feature_selection_model='RandomForest', 
    final_classifier_model='RandomForest', 
    use_smote=False,
    scaler='Standard',
    even_test_split=False
)

train_subject_nums, test_subject_nums = exp.debug()

# %%
