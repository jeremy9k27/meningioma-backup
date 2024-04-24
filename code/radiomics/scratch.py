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

im1 = sitk.ReadImage('data/preprocessing/output/7_COMPLETED_PREPROCESSED/6/6_Brainlab/12-AX_3D_T1_POST/6_Brainlab_12-AX_3D_T1_POST.nii.gz')
mask1 = sitk.ReadImage('data/segmentations/Segmentation 6.nii')

textures = collageradiomics.Collage(sitk.GetArrayFromImage(im1), sitk.GetArrayFromImage(mask1)).execute()

# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from LOOExperiment import LOOExperiment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

setup()

output_folder = 'data/radiomics_loo_debugging'

chr22q = LOOExperiment(
    prediction_task='Chr22q', 
    lambdas=[0.08, 0.12],
    output_dir=output_folder
)
chr22q_train_metrics_by_lambda, chr22q_test_metrics_by_lambda, chr22q_nonzero_coefs, chr22q_best_lambda = chr22q.loo_model() 

# %%
methyl = LOOExperiment(
    prediction_task='MethylationSubgroup', 
    lambdas=[0.08, 0.12],
    output_dir=output_folder
)
methyl_train_metrics_by_lambda, methyl_test_metrics_by_lambda, methyl_nonzero_coefs, methyl_best_lambda = methyl.loo_model(pmetric='Macro AUC') 

# %%
