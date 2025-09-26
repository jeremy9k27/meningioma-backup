# %% package imports
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup
from utils import clean_feature_names
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import seaborn as sns
sns.set_theme(style="whitegrid")



# more customized version of prep_data_for_loocv from preprocessing/utils.py that returns the subject ID numbers
def prep_data_for_loocv(features_file, pyrad = False, scaler_obj=StandardScaler()):
    # read in features and labels, merge
    data = pd.read_csv(features_file)
    data.columns = clean_feature_names(data.columns)
    data = data.dropna(axis=1, how='all').fillna(0)
    

    if pyrad:

        subjects = data['Patient Number_01']
        for i in range(1,25):
            if i <10:
                data = data.drop(columns=[f'Patient Number_0{i}'])
            else:
                data = data.drop(columns=[f'Patient Number_{i}'])

    else:
        subjects = data['Patient Number_01']
        for i in range(1,5):
            data = data.drop(columns=[f'Patient Number_0{i}'])


    X = data
    constant_feats = [col for col in X.columns if X[col].nunique() == 1]
    X = X.drop(columns=constant_feats)

    # scale data if specified
    if scaler_obj is not None:
        X = pd.DataFrame(scaler_obj.fit_transform(X), columns=X.columns)
    
    return X, subjects

# %%
# Read in radiomics features, labels, and subject ID numbers
radiomics_df, subs_r = prep_data_for_loocv(
    features_file='wide_features_pad.csv',
    pyrad = True
)


# enet
enet_df, subs_e = prep_data_for_loocv(
    features_file='wide_max_efficientnet_results.csv',
    pyrad = False
)



# Get the indices of those subject numbers occurring in both datasets
overlapping_subjects = list(set(subs_e).intersection(set(subs_r)))
overlapping_eidxs = np.where(subs_e.isin(overlapping_subjects))[0]
overlapping_ridxs = np.where(subs_r.isin(overlapping_subjects))[0]


# Filter the datasets to only include the overlapping subjects
enet_df = enet_df.iloc[overlapping_eidxs]
radiomics_df = radiomics_df.iloc[overlapping_ridxs]

print(len(enet_df))

# %%
def regression_analysis(predictors_df, outcomes_df, alpha=0.1, nonzero_threshold=0.99):
    '''
    Perform Lasso regression on each outcome in outcomes_df using predictors_df as the predictors.
    Return the R^2 values, the features selected, the coefficients, the number of non-zero coefficients, and the number of non-zero coefficients required to explain 80% of the variance.
    '''
    model = Ridge(alpha=alpha, max_iter=3000)

    rsquareds = []
    feats = []
    coefs = []
    num_nonzeros = []
    num_real_nonzeros = []
    score = 'n/a'
    num_feats = 'n/a'
    pbar = tqdm(range(outcomes_df.shape[1]), total=outcomes_df.shape[1], smoothing=0, desc=f'score = {score}, # feats (# exp 80% var) = {num_feats} ({num_feats})')
    for i in pbar:
        model.fit(predictors_df, outcomes_df.iloc[:, i])
        score = model.score(predictors_df, outcomes_df.iloc[:, i])
        rsquareds.append(score)

        num_nonzero = np.sum(model.coef_ != 0)
        num_nonzeros.append(num_nonzero)
        coef_sum = np.sum(np.abs(model.coef_))
        sorted_nonzero_coefs = np.sort(np.abs(model.coef_)/coef_sum)[::-1]
        coefs.append(sorted_nonzero_coefs)
        sorted_nonzero_coef_idxs = np.argsort(np.abs(model.coef_)/coef_sum)[::-1]
        sorted_nonzero_feats = predictors_df.columns[sorted_nonzero_coef_idxs[:num_nonzero]]
        feats.append(sorted_nonzero_feats)
        cumsum = np.cumsum(sorted_nonzero_coefs)
        nonzero_real = np.sum(cumsum <= nonzero_threshold)
        num_real_nonzeros.append(nonzero_real)

        pbar.set_description(f'score = {round(score, 2)}, # feats (# exp {round(nonzero_threshold*100)}% var) = {num_nonzero} ({nonzero_real})')
    
    # return rsquareds, feats, coefs, num_nonzeros, num_real_nonzeros
    return rsquareds, feats, coefs, num_nonzeros, num_real_nonzeros


def ridge_r2_analysis(predictors_df, outcomes_df, alpha=0.1):
    """
    Fits Ridge regression for each column in outcomes_df using predictors_df,
    and returns a list of R² values.
    """

    model = Ridge(alpha=alpha, max_iter=3000)
    rsquareds = []

    pbar = tqdm(range(outcomes_df.shape[1]), desc="R² computation", smoothing=0)
    for i in pbar:
        y = outcomes_df.iloc[:, i]
        model.fit(predictors_df, y)
        score = model.score(predictors_df, y)
        rsquareds.append(score)
        pbar.set_description(f"R² = {round(score, 2)}")
        if i == 200:
            break

    return rsquareds


'''
rsquareds = ridge_r2_analysis(predictors_df=enet_df, outcomes_df=radiomics_df, alpha=0.1)
stats1_df = pd.DataFrame({
    'rsquared': np.array(rsquareds).round(4),
}).sort_values(by=['rsquared'], ascending=[False])
stats1_counts = stats1_df.groupby(['rsquared']).size().reset_index(name='counts')
stats1_df.to_pickle('stats1_max_pad_ridge1.pkl')
'''


# %% run the regression analysis using collage features as predictors and radiomics features as 
rsquareds, feats, coefs, num_nonzeros, num_real_nonzeros = regression_analysis(predictors_df=enet_df, outcomes_df=radiomics_df, alpha=0.1, nonzero_threshold=0.99)
stats1_df = pd.DataFrame({
    'rsquared': np.array(rsquareds).round(2),
    'num_nonzeros': num_nonzeros,
    'num_real_nonzeros': num_real_nonzeros,
    'coeffs': coefs  # Include coefficient arrays directly
}).sort_values(by=['rsquared', 'num_real_nonzeros'], ascending=[False, True])


stats1_counts = stats1_df.groupby(['rsquared', 'num_real_nonzeros']).size().reset_index(name='counts')
stats1_df.to_pickle('stats1_max_ridge_with_coeffs.pkl')



# %% run the regression analysis using radiomics features as predictors and collage features as outcomes
rsquareds2, feats2, coefs2, num_nonzeros2, num_real_nonzeros2 = regression_analysis(predictors_df=radiomics_df, outcomes_df=enet_df, alpha=0.1, nonzero_threshold=0.99)
#rsquareds2, featsnum_nonzeros2, num_real_nonzeros2 = regression_analysis(predictors_df=radiomics_df, outcomes_df=enet_df, alpha=0.1, nonzero_threshold=0.99)


stats2_df = pd.DataFrame({
    'rsquared': np.array(rsquareds).round(2),
    'num_nonzeros': num_nonzeros,
    'num_real_nonzeros': num_real_nonzeros,
    'coeffs': coefs  # Include coefficient arrays directly
}).sort_values(by=['rsquared', 'num_real_nonzeros'], ascending=[False, True])
stats2_counts = stats2_df.groupby(['rsquared', 'num_real_nonzeros']).size().reset_index(name='counts')
stats2_df.to_pickle('stats2_max_ridge_with_coeffs.pkl')
