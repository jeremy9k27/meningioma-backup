# %% Package imports
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
from utils import clean_feature_names

setup()

data_dir = 'data/radiomics/'

# Read in the data
not_v = pd.read_csv(f'{data_dir}/comparison_NOTvirginia/features_wide.csv')
v = pd.read_csv(f'{data_dir}/comparison_virginia/features_wide.csv')

# Clean up the column names
not_v.columns = clean_feature_names(not_v.columns)
v.columns = clean_feature_names(v.columns)

# Drop columns with only one unique value
not_v = not_v.drop(columns=not_v.columns[not_v.nunique() == 1])
v = v.drop(columns=v.columns[v.nunique() == 1])

# Select overlapping columns from each dataset
overlap_cols = not_v.columns.intersection(v.columns)
v = v[overlap_cols]
not_v = not_v[overlap_cols]

# Replace NaNs with 0
v = v.fillna(0)
not_v = not_v.fillna(0)

# Element-wise subtraction
diff = abs(v - not_v) / (abs(v) + 0.0000000001) 
diff = diff.drop(columns=['Subject Number'])

# Rearrange the columns by the column mean
diff = diff.reindex(diff.median().sort_values(ascending=False).index, axis=1)

# %% Box and whisker plots
def make_boxplot(cols=range(10)):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=diff.iloc[:, cols], orient='h', showfliers=False)
    plt.title('Feature Sensitivity to Annotation abs(V - R)/abs(V)')
    plt.show()

make_boxplot(range(10))
make_boxplot(range(10, 20))
make_boxplot(range(20, 30))
make_boxplot(range(670, 680))

# %% Line graph
median_df = pd.DataFrame(diff.median().sort_values(ascending=False), columns=['Median']).reset_index()
median_df = median_df.rename(columns={'index': 'Feature'})

sns.set_theme(style='whitegrid')
plt.figure(figsize=(24, 8))
ax = sns.lineplot(data=median_df, x='Feature', y='Median', marker='o')
plt.title('Median Difference in Feature Values')
plt.xlabel('Feature')
plt.ylabel('Median Difference Values abs(V - R)/abs(V)')
plt.xticks(rotation=45)
ax.set_yscale('log')
plt.show()

# %%
# Save the most and least sensitive features
if not os.path.exists('data/peter_exp'):
    os.makedirs('data/peter_exp')
median_df.head(75).to_csv('data/peter_exp/most_sensitive_features.csv', index=False)
median_df.sort_values('Median').head(275).to_csv('data/peter_exp/least_sensitive_features.csv', index=False)
# %%
