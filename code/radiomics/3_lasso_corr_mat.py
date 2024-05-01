# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from utils import prep_data_for_loocv, plot_corr_matrix
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
setup()

# %%
def get_nonzero_feats(exp):
    feat_dict = {}
    if exp.n_classes > 2:
        for k, biomarker in enumerate(exp.class_ids):
            coefs = pd.DataFrame(exp.coef[:, k, :], columns=exp.feat_names)
            zero_feats = [col for col in coefs.columns if (coefs[col] == 0.).all()]
            nonzero_coefs = coefs.drop(columns=zero_feats).T
            nonzero_coefs['Absolute Sum'] = nonzero_coefs.abs().sum(axis=1)
            nonzero_coefs = nonzero_coefs.sort_values(by='Absolute Sum', ascending=False)
            nonzero_coefs['Prop Var Exp'] = nonzero_coefs['Absolute Sum'] / nonzero_coefs['Absolute Sum'].sum()                
            nonzero_coefs['Cum Var Exp'] = nonzero_coefs['Prop Var Exp'].cumsum()
            feat_dict[biomarker] = nonzero_coefs
    else:
        coefs = pd.DataFrame(exp.coef.squeeze(), columns=exp.feat_names)
        zero_feats = [col for col in coefs.columns if (coefs[col] == 0.).all()]
        nonzero_coefs = coefs.drop(columns=zero_feats).T
        nonzero_coefs['Absolute Sum'] = nonzero_coefs.abs().sum(axis=1)
        nonzero_coefs = nonzero_coefs.sort_values(by='Absolute Sum', ascending=False)
        nonzero_coefs['Prop Var Exp'] = nonzero_coefs['Absolute Sum'] / nonzero_coefs['Absolute Sum'].sum()
        nonzero_coefs['Cum Var Exp'] = nonzero_coefs['Prop Var Exp'].cumsum()
        feat_dict['Chr22q'] = nonzero_coefs

    return feat_dict

def plot_correlations(task='MethylationSubgroup'):
    exp = joblib.load(f'data/OLD_classic_loo/{task}/exp.pkl')
    feats_dict = get_nonzero_feats(exp)
    X, _ = prep_data_for_loocv(outcome=task, scaler_obj=StandardScaler())
    for biomarker in exp.class_ids:
        plot_corr_matrix(X[feats_dict[biomarker].index.to_list()], outcome=biomarker)

# Correlation matrix for MethylationSubgroup + Chr22q task
# %%
plot_correlations()
# %%
