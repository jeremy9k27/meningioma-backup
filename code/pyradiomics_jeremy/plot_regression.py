# %%
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
import matplotlib.pyplot as plt



def histplot_rsq(df, title):
    '''
    Plots a histogram of R² values from the given DataFrame.

    Parameters:
    df
    title
    '''
    print(df['rsquared'])
    plt.figure(figsize=(8, 5))
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    ax = sns.histplot(
        data=df,
        x='rsquared',
        bins=20,
        kde=False,
        color='steelblue',
        edgecolor='black'
    )

    ax.xaxis.grid(True, which='minor', linewidth=0.25)
    ax.yaxis.grid(True, which='minor', linewidth=0.25)
    sns.despine(left=True, bottom=True)

    ax.set_title(title)
    ax.set_xlabel(r'$R^2$ Values')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0.0, 1.1)
    plt.tight_layout()
    plt.show()


statsdf = pd.read_pickle('stats1_max_pad_ridge1.pkl')
histplot_rsq(statsdf, title='Regressing Pyrad features on Enet features (max pooling) with Ridge')


#%%

def coeff_dist(coeffs, title):
    # Plot histogram
    coeffs = np.concatenate(coeffs.values)
    plt.figure(figsize=(8, 5))
    sns.histplot(coeffs, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Coefficient Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

statsdf = pd.read_pickle('stats1_max_ridge_with_coeffs.pkl')
coeffs = statsdf['coeffs']
coeff_dist(coeffs, title = "Regressing Enet (max pool) with Pyrad")

statsdf = pd.read_pickle('stats1_avg_ridge_with_coeffs.pkl')
coeffs = statsdf['coeffs']
coeff_dist(coeffs, title = "Regressing Enet (avg pool) with Pyrad")
# %%


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





stats2_df = pd.read_pickle('stats1_max_pad_ridge1.pkl')

stats_counts = stats2_df.groupby(['rsquared', 'num_real_nonzeros']).size().reset_index(name='counts')



relplot(stats_counts, title='Regressing on Enet features (avg pooling) on Pyrad features with Ridge')
# %%
