# # %%
# import collageradiomics
# import SimpleITK as sitk
# import numpy as np
# import os
# import sys
# import pandas as pd
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# from preprocessing.utils import setup

# setup()

# im1 = sitk.ReadImage('data/preprocessing/output/7_COMPLETED_PREPROCESSED/6/6_Brainlab/12-AX_3D_T1_POST/6_Brainlab_12-AX_3D_T1_POST.nii.gz')
# mask1 = sitk.ReadImage('data/segmentations/Segmentation 6.nii')

# textures = collageradiomics.Collage(sitk.GetArrayFromImage(im1), sitk.GetArrayFromImage(mask1)).execute()

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
import joblib

setup()

output_folder = 'data/radiomics_par_loo_debug'

chr22q = LOOExperiment(
    prediction_task='Chr22q', 
    lambdas=[0.08, 0.12],
    output_dir=output_folder,
    use_smote=True
)

# %%
chr22q_train_metrics_by_lambda, chr22q_test_metrics_by_lambda, chr22q_nonzero_coefs, chr22q_best_lambda = chr22q.par_loo_model() 
joblib.dump(chr22q, f'{output_folder}/{task}/chr22q.pkl')
# %%
methyl = LOOExperiment(
    prediction_task='MethylationSubgroup', 
    lambdas=[0.08, 0.12],
    output_dir=output_folder,
    use_smote=True
)
methyl_train_metrics_by_lambda, methyl_test_metrics_by_lambda, methyl_nonzero_coefs, methyl_best_lambda = methyl.par_loo_model(pmetric='Macro AUC') 

# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
import joblib
import numpy as np

setup()

exp = joblib.load('data/classic_loo/Chr22q/exp.pkl')
exp2 = joblib.load('data/classic_loo/MethylationSubgroup/exp.pkl')
# %%
import numpy as np
np.arange(0.2, 2.0, 0.2)

# %%
import numpy as np

def split_array(array, value):
    # Find the index of the specified value in the array
    index = np.where(array == value)[0][0]
    
    # Create one array with the specified value
    array_with_value = array[index:index+1]
    
    # Create another array with the remaining values
    remaining_array = np.concatenate([array[:index], array[index+1:]])
    
    return array_with_value, remaining_array

# %%
# Example usage
n = 10
array = np.arange(n)
value = 5
array_with_value, remaining_array = split_array(array, value)

print("Array with value:", array_with_value)
print("Remaining array:", remaining_array)
# %%
