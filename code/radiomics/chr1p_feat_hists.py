# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import prep_data_for_loocv

setup()

data_dir = 'results/classic_loo_smoothed_radiomics_fine_5-15-24/Chr1p' # 'results/LOO_normalized_radiomics_6-24-24_fine_smallest/Chr1p'
coef_df = pd.read_csv(f'{data_dir}/Chr1p_coefs.csv')

# %%
X, y = prep_data_for_loocv(
    features_file='data/radiomics/features8_smoothed/features_wide.csv', # 'data/4c_radiomics_adjusted_w_medians_and_mads/features_wide.csv', 
    outcome='Chr1p', 
    scaler_obj=StandardScaler()
)
X['y'] = y
feats_of_interest = coef_df.iloc[0:6, 0].to_numpy()

# %%
for feat in feats_of_interest:
    sns.displot(data=X, x=feat, hue='y', multiple='stack', bins=40)
# %%
