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
from sklearn.linear_model import Lasso, LinearRegression
import seaborn as sns
sns.set_theme(style="whitegrid")

setup()

outcome = 'MethylationSubgroup' # 'Chr22q'

# more customized version of prep_data_for_loocv from preprocessing/utils.py that returns the subject ID numbers
def prep_data_for_loocv(features_file, labels_file='data/labels/MeningiomaBiomarkerData.csv', outcome='MethylationSubgroup', scaler_obj=StandardScaler()):
    # read in features and labels, merge
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)
    labels = labels.dropna(subset=[outcome])
    labels = labels[labels['Subject Number'].isin(features['Subject Number'])]
    data = features.merge(labels, on='Subject Number')
    data.columns = clean_feature_names(data.columns)
    data = data.dropna(axis=1, how='all').fillna(0)
    subjects = data['Subject Number']
    X = data.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
    constant_feats = [col for col in X.columns if X[col].nunique() == 1]
    X = X.drop(columns=constant_feats)
    y = data[outcome].values.astype(int)

    # scale data if specified
    if scaler_obj is not None:
        X = pd.DataFrame(scaler_obj.fit_transform(X), columns=X.columns)
    
    return X, y, subjects

# %%
# Read in collage features, labels, and subject ID numbers
collage_df, y_c, subs_c = prep_data_for_loocv(
    features_file='data/collage_sparse_small_windows/windowsize-5_binsize-32_summary_22nansfilled.csv',
    outcome=outcome
)

# Read in radiomics features, labels, and subject ID numbers
radiomics_df, y_r, subs_r = prep_data_for_loocv(
    features_file='data/radiomics/features8_smoothed/features_wide.csv',
    outcome=outcome
)

# Get the indices of those subject numbers occurring in both datasets
overlapping_subjects = list(set(subs_c).intersection(set(subs_r)))
overlapping_cidxs = np.where(subs_c.isin(overlapping_subjects))[0]
overlapping_ridxs = np.where(subs_r.isin(overlapping_subjects))[0]

# Filter the datasets to only include the overlapping subjects
collage_df = collage_df.iloc[overlapping_cidxs]
radiomics_df = radiomics_df.iloc[overlapping_ridxs]
y_c = y_c[overlapping_cidxs]
y_r = y_r[overlapping_ridxs]

# mi = mutual_info_regression(collage_df, radiomics_df.iloc[:, 0])

noise_data = np.random.normal(loc=0, scale=1, size=radiomics_df.shape)
df_noise = pd.DataFrame(noise_data, columns=[f'Col{i+1}' for i in range(radiomics_df.shape[1])])

# %%
def regression_analysis(predictors_df, outcomes_df, alpha=0.1, nonzero_threshold=0.99):
    '''
    Perform Lasso regression on each outcome in outcomes_df using predictors_df as the predictors.
    Return the R^2 values, the features selected, the coefficients, the number of non-zero coefficients, and the number of non-zero coefficients required to explain 80% of the variance.
    '''
    model = Lasso(alpha=alpha, max_iter=3000)

    rsquareds = []
    # feats = []
    # coefs = []
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
        # coefs.append(sorted_nonzero_coefs)
        # sorted_nonzero_coef_idxs = np.argsort(np.abs(model.coef_)/coef_sum)[::-1]
        # sorted_nonzero_feats = predictors_df.columns[sorted_nonzero_coef_idxs[:num_nonzero]]
        # feats.append(sorted_nonzero_feats)
        cumsum = np.cumsum(sorted_nonzero_coefs)
        nonzero_real = np.sum(cumsum <= nonzero_threshold)
        num_real_nonzeros.append(nonzero_real)

        pbar.set_description(f'score = {round(score, 2)}, # feats (# exp {round(nonzero_threshold*100)}% var) = {num_nonzero} ({nonzero_real})')
    
    # return rsquareds, feats, coefs, num_nonzeros, num_real_nonzeros
    return rsquareds, num_nonzeros, num_real_nonzeros

# %%
noise_rsquareds, noise_num_nonzeros, noise_num_real_nonzeros = regression_analysis(predictors_df=radiomics_df, outcomes_df=df_noise, alpha=0.1, nonzero_threshold=0.99)

# %% baseline regression analysis using radiomics features as BOTH predictors and outcomes, to see how Lasso behaves
# baseline_rsquareds, baseline_feats, baseline_coefs, baseline_num_nonzeros, baseline_num_real_nonzeros = regression_analysis(predictors_df=radiomics_df, outcomes_df=radiomics_df, alpha=0.1, nonzero_threshold=0.99)
baseline_rsquareds, baseline_num_nonzeros, baseline_num_real_nonzeros = regression_analysis(predictors_df=radiomics_df, outcomes_df=radiomics_df, alpha=0.1, nonzero_threshold=0.99)

baseline_stats_df = pd.DataFrame({
    'rsquared': np.array(baseline_rsquareds).round(2),
    'num_nonzeros': baseline_num_nonzeros,
    'num_real_nonzeros': baseline_num_real_nonzeros
}).sort_values(by=['rsquared', 'num_real_nonzeros'], ascending=[False, True])
baseline_stats_counts = baseline_stats_df.groupby(['rsquared', 'num_real_nonzeros']).size().reset_index(name='counts')

theoretical_counts = pd.DataFrame({'rsquared': [1.0, 0.0], 'num_real_nonzeros': [0, 0], 'counts': [radiomics_df.shape[1], 0]})

# %%
def relplot(df, title='Dataset Comparison'):
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    if len(df) == 2:
        g = sns.relplot(
            data=df,
            x="rsquared", y="num_real_nonzeros",
            size="counts", sizes=(10, 800),
        )
        g._legend.remove()
    else:
        g = sns.relplot(
            data=df,
            x="rsquared", y="num_real_nonzeros",
            hue="rsquared", size="counts",
            palette=cmap, sizes=(10, 200),
        )

    # g.ax.set_xlim(0.5, 1.05)
    # g.ax.set_ylim(-1, 15)
    g.ax.invert_xaxis()
    g.ax.invert_yaxis()

    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    g.despine(left=True, bottom=True)

    g.ax.set_title(title)
    g.ax.set_xlabel(r'$R^2$ Values')
    g.ax.set_ylabel(r'# Features w/cum. var. $\leq$ 99%')

# %%
relplot(baseline_stats_counts, title='Regressing pyradiomics features on themselves in practice')
relplot(theoretical_counts, title='What regressing a dataset on itself should look like in theory')

# %% run the regression analysis using collage features as predictors and radiomics features as outcomes
rsquareds, feats, coefs, num_nonzeros, num_nonzero80s = regression_analysis(predictors_df=collage_df, outcomes_df=radiomics_df, alpha=0.1, nonzero_threshold=0.99)

# %% run the regression analysis using radiomics features as predictors and collage features as outcomes
# rsquareds2, feats2, coefs2, num_nonzeros2, num_real_nonzeros2 = regression_analysis(predictors_df=radiomics_df, outcomes_df=collage_df, alpha=0.1, nonzero_threshold=0.99)
rsquareds2, num_nonzeros2, num_real_nonzeros2 = regression_analysis(predictors_df=radiomics_df, outcomes_df=collage_df, alpha=0.1, nonzero_threshold=0.99)

# %%
stats2_df = pd.DataFrame({
    'rsquared': np.array(rsquareds2).round(2),
    'num_nonzeros': num_nonzeros2,
    'num_real_nonzeros': num_real_nonzeros2
}).sort_values(by=['rsquared', 'num_real_nonzeros'], ascending=[False, True])
stats2_counts = stats2_df.groupby(['rsquared', 'num_real_nonzeros']).size().reset_index(name='counts')

noise_stats_df = pd.DataFrame({
    'rsquared': np.array(noise_rsquareds).round(2),
    'num_nonzeros': noise_num_nonzeros,
    'num_real_nonzeros': noise_num_real_nonzeros
}).sort_values(by=['rsquared', 'num_real_nonzeros'], ascending=[False, True])
noise_counts = noise_stats_df.groupby(['rsquared', 'num_real_nonzeros']).size().reset_index(name='counts')

# %%
relplot(stats2_counts, title='Regressing CoLlAGe features on PyRadiomics features')
relplot(noise_counts, title='Regressing random noise on PyRadiomics features')
# %%
