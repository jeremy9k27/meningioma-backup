# File: 2a_feature_selection.py
# Date: 03/16/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description:
#
# This script relies on the following file(s) as inputs:
#   *
#   *
#
# This script generates the following file(s) as outputs:
#   *
#   *

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.feature_selection import RFE, RFECV, mutual_info_classif, chi2, f_classif, SelectKBest, SelectFpr, SelectFdr, SelectFwe
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import get_data, plot_corr_matrix
from preprocessing.utils import setup
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import joblib

setup()

# Researcher specified global constants
FEAT_FILE = 'data/radiomics/features5/features_wide.csv'
OUTCOME = 'MethylationSubgroup' # 'MethylationSubgroup' or 'Chr1p' or 'Chr22q' or 'Chr9p' or 'TERT'
TEST_SIZE = 21
SEED = 5
USE_SMOTE = False
SCALER = None # 'Standard' or 'MinMax' or None
EVEN_TEST_SPLIT = False
OUTPUT_DIR = None # f'data/radiomics/evaluations5/{OUTCOME}_TestSize-{TEST_SIZE}_Seed-{SEED}_SMOTE-{USE_SMOTE}_EvenTestSplit-{EVEN_TEST_SPLIT}_Scaler-{SCALER}'

# Auto-generated global constants
if OUTCOME == 'MethylationSubgroup':
    CLASS_IDS = ['Merlin Intact', 'Immune Enriched', 'Hypermetabolic']
else:
    CLASS_IDS = ['Intact', 'Loss']
N_CLASSES = len(CLASS_IDS)
if USE_SMOTE: assert SCALER is not None, "Must specify a scaler when using SMOTE"
if SCALER == 'Standard':
    SCALER_OBJ = StandardScaler()
elif SCALER == 'MinMax':
    SCALER_OBJ = MinMaxScaler()
else:
    SCALER_OBJ = None
if OUTPUT_DIR is not None and not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def univariate_feature_selection(X, y, method=mutual_info_classif, method_name='MI'):
    """
    Perform univariate feature selection using the specified method and return a dataframe with the feature scores and ranks.
    """
    selector = SelectKBest(method, k='all')
    selector = selector.fit(X, y)
    df = pd.DataFrame({'feat': X.columns, f'{method_name}': selector.scores_}).sort_values(by=f'{method_name}', ascending=False)
    df[f'{method_name}_rank'] = df[f'{method_name}'].rank(ascending=False)
    return df

def plot_feat_ranks_scatter(df, sort_val_key='mean_rank', top_k=128, categorical=False, filename=None):
    """
    Plots a scatter plot of feature ranks for the top_k features based on the sort_val_key. 
    If categorical is True, the scatter points will be colored based on the method used to calculate the rank. 
    Otherwise, the scatter points will be colored based on the sort_val_key.
    """
    rank_cols = [col for col in df.columns if '_rank' in col]
    sort_val = [col for col in rank_cols if sort_val_key in col][0]
    df_ranks = df[['feat'] + rank_cols].sort_values(by=sort_val, ascending=True)
    df_ranks = df_ranks.head(top_k)
    melted_df = df_ranks.melt(id_vars=['feat', sort_val], value_vars=rank_cols, var_name='Method', value_name='rank_value')

    if categorical:
        # Unique MI_rank for color mapping
        unique_mi_ranks = melted_df['Method'].unique()
        colors = plt.cm.get_cmap('viridis', len(unique_mi_ranks))
    else:
        # Normalize sort_val for color mapping
        norm = plt.Normalize(df_ranks[sort_val].min(), df_ranks[sort_val].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # increase x positions by 2 to separate the scatter points
    x_positions = np.arange(len(df_ranks['feat'])) * 2

    # Create a scatter plot for each feature
    for i, (feat, x_pos) in enumerate(zip(df_ranks['feat'], x_positions)):
        feat_df = melted_df[melted_df['feat'] == feat]
        
        if categorical:
            # Plot scatter points with color based on MI_rank
            for mi_rank in unique_mi_ranks:
                subset = feat_df[feat_df['Method'] == mi_rank]
                ax.scatter([x_pos] * len(subset), subset['rank_value'], color=colors(np.where(unique_mi_ranks == mi_rank)[0][0]), edgecolor='k', label=mi_rank if i == 0 else "")
        else:
            # Plot scatter points
            ax.scatter([x_pos] * len(feat_df), feat_df['rank_value'], color=sm.to_rgba(feat_df[sort_val].iloc[0]), edgecolor='k')
    
    # Customizing the plot
    ax.xaxis.set_ticks([])
    ax.set_xlabel('Feature')
    ax.set_ylabel('Rank Value')
    ax.set_title('Scatter Plot of Ranks by Feature')

    # Legend
    if categorical:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Colorbar
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Overall mean rank')
    
    plt.tight_layout()
    if OUTPUT_DIR is not None and filename is not None: 
        plt.savefig(f'{OUTPUT_DIR}/{filename}')
    else:
        plt.show()
    plt.close()

def plot_feat_ranks_line(df, sort_val_key = 'MImean_rank', std_key='MIstd', top_k=128, filename=None):
    """
    Plots a line graph of feature ranks for the top_k features based on the sort_val_key, and confidence intervals based on the std_key.
    """
    df_sorted = df.sort_values(by=sort_val_key, ascending=True).head(top_k)
    plt.figure(figsize=(10, 6))  # Set the figure size as desired
    plt.plot(df_sorted['feat'], df_sorted[sort_val_key], label='Mean Rank')  # Line graph of means

    # Confidence intervals (mean ± std)
    plt.fill_between(df_sorted['feat'], df_sorted[sort_val_key] - df_sorted[std_key], df_sorted[sort_val_key] + df_sorted[std_key], color='gray', alpha=0.2, label='Confidence Interval')

    plt.gca().set_xticks([])
    plt.xlabel('Feature')
    plt.ylabel('Mean Rank')
    plt.title('Line Graph of Features by Mean with Confidence Intervals')
    plt.legend()
    plt.tight_layout()
    if OUTPUT_DIR is not None and filename is not None:
        plt.savefig(f'{OUTPUT_DIR}/{filename}')
    else:
        plt.show()
    plt.close()

def get_uni_sets(df, top_k=1024):
    """
    Returns the feature sets for the top_k features from each column in df ending in '_rank', the union of all features, and the intersection of all features.
    """
    rank_cols = [col for col in df.columns if '_rank' in col]
    feat_sets = {}
    for col in rank_cols:
        feat_sets[col] = set(df.sort_values(by=col, ascending=True).head(top_k)['feat'])
    union = set.union(*feat_sets.values())
    intersection = set.intersection(*feat_sets.values())

    return feat_sets, union, intersection

def plot_rfecv_results(rfecv):
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(1, n_scores + 1),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()

def feature_selection(X, y):
    # Step 1: Univariate feature selection
    if os.path.exists(f"{OUTPUT_DIR}/overall_uni_feats.csv"):
        overall_uni_df = pd.read_csv(f"{OUTPUT_DIR}/overall_uni_feats.csv")
    else:
        print("Step 1/3: Performing univariate feature selection...")
        univariate_df = None

        # We will start with mutual information, which needs multiple runs to combat its sensitivity to random seed
        # We will run MI 20 times and take the mean of the scores.
        num_MI_runs = 20
        for i in tqdm(range(num_MI_runs)):
            df = univariate_feature_selection(X, y, method=lambda X, y: mutual_info_classif(X=X, y=y, random_state=i), method_name=f'MI{i}')
            if univariate_df is None:
                univariate_df = df
            else:
                univariate_df = univariate_df.merge(df, on='feat', how='outer')
        # Now we have the MI score for each feature across all runs, we can take the mean & std dev and rank them
        # We will rank them based on the mean score alone. The std dev is for visualizing the spread of the scores.
        univariate_df['MImean'] = univariate_df[[f'MI{i}' for i in range(num_MI_runs)]].mean(axis=1)
        univariate_df['MImean_rank'] = univariate_df['MImean'].rank(ascending=False)
        univariate_df['MIstd'] = univariate_df[[f'MI{i}_rank' for i in range(num_MI_runs)]].std(axis=1)
        # Let's plot the top 512 features by mean MI score
        plot_feat_ranks_scatter(univariate_df, top_k=512)
        plot_feat_ranks_line(univariate_df, top_k=512)

        # Now we will perform univariate feature selection using Chi2 and ANOVA
        methods = [chi2, f_classif]
        method_names = ['Chi2', 'ANOVA']
        # We need to scale the features for these methods - to preserve the original distribution of the data we will use MinMaxScaler
        X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
        for method, method_name in zip(methods, method_names):
            df = univariate_feature_selection(X_scaled, y, method=method, method_name=method_name)
            univariate_df = univariate_df.merge(df, on='feat', how='outer')
        overall_uni_df = univariate_df[['feat', 'MImean_rank', 'Chi2_rank', 'ANOVA_rank']]
        # Chi2 and ANOVA may return NaNs for some features that don't have sufficient information, we will fill these with the middle rank (num features // 2)
        # before calculating the average rank across all univariate methods
        overall_uni_df.fillna(overall_uni_df.shape[0]//2, inplace=True)
        overall_uni_df['avg_rank'] = overall_uni_df[[col for col in overall_uni_df.columns if '_rank' in col]].mean(axis=1)
        overall_uni_df['std'] = overall_uni_df[[col for col in overall_uni_df.columns if '_rank' in col]].std(axis=1)
        overall_uni_df.sort_values(by='avg_rank', ascending=True, inplace=True)
        if OUTPUT_DIR is not None: overall_uni_df.to_csv(f'{OUTPUT_DIR}/overall_uni_feats.csv', index=False)

        # Now we can plot the sensitivity of the features to the different univariate methods
        plot_feat_ranks_scatter(overall_uni_df, sort_val_key='avg_rank')
        plot_feat_ranks_scatter(overall_uni_df, sort_val_key='avg_rank', categorical=True)
        plot_feat_ranks_line(overall_uni_df, sort_val_key='avg_rank', std_key='std')

    # Get the top 1024 features from each univariate method
    _, all_uni_features, robust_uni_features = get_uni_sets(overall_uni_df, top_k=1024)
    X_reduced = X[all_uni_features]
    # univariate_selected_features = X_reduced.columns
    print(f"Number of robust features (common across all univariate selection methods): {len(robust_uni_features)}")
    print(f"Number of unique features across all methods: {len(all_uni_features)}")

    # UNCOMMENT THIS BLOCK LATER
    # # Step 2: Gridsearch for a RF classifier to use during recursive feature elimination in next step
    # print("Step 2/3: Performing gridsearch for RF classifier to use during RFE step...")
    # # check if f"{OUTPUT_DIR}/gridsearch1_feature_selection.joblib" exists, if so, load it and skip gridsearch
    # if os.path.exists(f"{OUTPUT_DIR}/gridsearch1_feature_selection.joblib"):
    #     gs = joblib.load(f"{OUTPUT_DIR}/gridsearch1_feature_selection.joblib")
    #     print("Gridsearch results loaded from file.")
    # else:
    #     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    #     estimator = RandomForestClassifier()
    #     params = {
    #         'n_estimators': [25, 50, 75, 100, 200],
    #         'criterion': ['gini', 'entropy', 'log_loss'],
    #         'max_depth': [None, 10, 20],
    #         'min_samples_split': [2, 4, 6],
    #         'max_features': ['sqrt', 'log2', None],
    #         'bootstrap': [True],
    #         'class_weight': ['balanced', 'balanced_subsample']
    #     }

    #     gs = GridSearchCV(
    #         estimator=estimator,
    #         param_grid=params,
    #         scoring='f1_macro',
    #         refit=False,
    #         n_jobs=4,
    #         cv=cv,
    #         return_train_score=True,
    #         verbose=10
    #     )

    #     gs.fit(X_reduced, y)

    #     if OUTPUT_DIR is not None: joblib.dump(gs, f'{OUTPUT_DIR}/gridsearch1_feature_selection.joblib')

    # print("\tBest parameters from gridsearch: ", gs.best_params_)
    # print("\tBest score from gridsearch: ", gs.best_score_)
    # UNCOMMENT THIS BLOCK LATER

    # Step 3: Cross-validation and recursive feature elimination
    print("Step 3/3: Performing cross-validated recursive feature elimination...")
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    # feature_ranks = []
    # clf = RandomForestClassifier(**gs.best_params_, random_state=SEED)
    best_params = {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'log_loss', 'max_depth': 10, 'max_features': None, 'min_samples_split': 6, 'n_estimators': 100}
    clf = RandomForestClassifier(**best_params, random_state=SEED)
    rfecv = RFECV(estimator=clf, min_features_to_select=1, step=len(all_uni_features)//32, cv=kf, scoring='f1_macro', verbose=3, n_jobs=4)
    rfecv.fit(X_reduced, y)

    # for train_index, test_index in tqdm(kf.split(X_reduced, y), total=kf.get_n_splits()):
    #     X_train, X_test = X_reduced[train_index], X_reduced[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
        
    #     rf = RandomForestClassifier(**gs.best_params_, random_state=SEED)
    #     rfe = RFE(estimator=rf, n_features_to_select=1, step=len(all_uni_features)//32, verbose=0)
    #     rfe.fit(X_train, y_train)
        
    #     # Rank features
    #     feature_ranks.append(rfe.ranking_)

    # feature_ranks = np.stack(feature_ranks, axis=1)
    # average_ranks = np.mean(feature_ranks, axis=1)
    # Step 3: Select the 32 best features based on average rank
    # best_features_indices = np.argsort(average_ranks)[:top_k]
    # final_selected_features = univariate_selected_features[best_features_indices]
    # final_selected_feature_ranks = feature_ranks[best_features_indices, :]

    # # Step 4: Plot correlation matrix of final selected features if specified
    # plot_corr_matrix(X[final_selected_features], outcome=OUTCOME, test_size=TEST_SIZE, output_dir=OUTPUT_DIR)

    return overall_uni_df, rfecv

def run_model(X_train_df, y_train, X_test_df, y_test, features):
    # Prep data for classifier
    X_train = X_train_df[features]
    if USE_SMOTE:
        print(f"Before SMOTE: X_train: {X_train.shape}, y_train: {y_train.shape}")
        X_train, y_train = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
        print(f"After SMOTE: X_train: {X_train.shape}, y_train: {y_train.shape}")

    X_test = X_test_df[features]

    # Train classifier
    print("Final gridsearch for the final classifier...")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    estimator = RandomForestClassifier()
    params = {
        'n_estimators': [25, 50, 75, 100, 200],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring='f1_macro',
        refit=True,
        n_jobs=-1,
        cv=cv,
        return_train_score=True,
        verbose=10
    )

    # 6d. Run gridsearch
    gs.fit(X_train, y_train)
    joblib.dump(gs, f'{OUTPUT_DIR}/gridsearch2_model.joblib')

    # 6e. Save results
    results = gs.cv_results_
    keys = [k for k, v in results.items() if k.startswith('split')]
    for x in keys:
        del results[x]

    results_df = pd.DataFrame().from_dict(results)
    print("\nBEST PARAMS:\n")
    print(gs.best_params_)

    classifier = gs.best_estimator_
    train_probs = classifier.predict_proba(X_train)
    test_probs = classifier.predict_proba(X_test)

    return train_probs, test_probs, y_train, y_test, results_df, gs.best_params_, gs.best_score_, classifier

def plot_binary_metrics(train_probs, test_probs, y_train, y_test, use_test=True, save_fig=True):
    """
    Plot ROC curve, confusion matrix, and metrics table for binary classification tasks
    """
    # whether to use the test or training set results
    if not use_test:
        y_test = y_train
        test_probs = train_probs

    # Plot 1/3: ROC curve:
    fpr, tpr, _ = roc_curve(y_test, test_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='tab:orange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='tab:blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {OUTCOME} (Train/test split: {y_train.shape[0]}/{TEST_SIZE})')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{OUTPUT_DIR}/roc_curve.png')
    else:
        plt.show()
    plt.close()

    # Plot 2/3: Confusion matrix
    predicted_labels = np.argmax(test_probs, axis=1)
    conf_matrix = confusion_matrix(y_test, predicted_labels)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = accuracy_score(y_test, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='viridis', cbar=False, xticklabels=CLASS_IDS, yticklabels=CLASS_IDS)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix: {OUTCOME} (Train/test split: {y_train.shape[0]}/{TEST_SIZE})\nOverall Accuracy = {accuracy*100:.2f}%')
    
    if save_fig:
        plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png')
    else:
        plt.show()
    plt.close()

    # Plot 3/3: Metrics table
    metrics = {
        'AUC': roc_auc,
        'F1 Score': f1_score(y_test, predicted_labels, average='binary'), # or weighted
        'Precision': precision_score(y_test, predicted_labels, average='binary'), # or weighted
        'Recall (Sensitivity)': recall_score(y_test, predicted_labels, average='binary'), # or weighted
        'Specificity': tn / (tn + fp),
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_accuracy_score(y_test, predicted_labels),
        'MCC': matthews_corrcoef(y_test, predicted_labels)
    }

    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])

    _, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc = 'center', loc='center')
    if save_fig:
        plt.savefig(f'{OUTPUT_DIR}/metrics_table.png')
    else:
        plt.show()
    plt.close()

def plot_gs_metrics(train_probs, test_probs, y_train, y_test, use_test=True, class_ids=CLASS_IDS, save_fig=True):
    """Plot ROC curve and confusion matrix for multiclass classification tasks"""
    # whether to use the test or training set results
    if not use_test:
        y_test = y_train
        test_probs = train_probs
    
    # Binarize multiclass labels
    y_test = label_binarize(y_test, classes=np.arange(N_CLASSES))

    fpr, tpr, roc_auc = dict(), dict(), dict()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), test_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], test_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(N_CLASSES):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= N_CLASSES

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot 1/2: ROC curve
    _, ax = plt.subplots(figsize=(9, 9))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(sns.color_palette())
    for i, color, class_id in zip(range(N_CLASSES), colors, class_ids):
        RocCurveDisplay.from_predictions(
            y_test[:, i],
            test_probs[:, i],
            name=f"ROC curve for {class_id}",
            color=color,
            ax=ax,
            plot_chance_level=(i == N_CLASSES - 1),
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"One-vs-Rest ROC Curves: {OUTCOME} (Train/test split: {y_train.shape[0]}/{TEST_SIZE})",
    )

    if save_fig: 
        plt.savefig(f'{OUTPUT_DIR}/roc_curve.png')
    else:
        plt.show()

    plt.close()

    # Plot 2/2: Confusion matrix
    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(test_probs, axis=1)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='viridis', cbar=False, xticklabels=class_ids, yticklabels=class_ids)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix: {OUTCOME} (Train/test split: {y_train.shape[0]}/{TEST_SIZE})\nOverall Accuracy = {accuracy*100:.2f}%')
    
    if save_fig:
        plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png')
    else:
        plt.show()
    
    plt.close()

#%%
X_train_df, y_train, X_test_df, y_test = get_data(features_file=FEAT_FILE, outcome=OUTCOME, test_size=TEST_SIZE, seed=SEED, even_test_split=EVEN_TEST_SPLIT, scaler_obj=SCALER_OBJ, output_dir=OUTPUT_DIR)

overall_uni_df, rfecv = feature_selection(X_train_df, y_train)
# final_selected_features, final_selected_feature_ranks = feature_selection(X_train_df, y_train)

# %%
train_probs, test_probs, y_train, y_test, results_df, best_params, best_score, classifier = run_model(X_train_df, y_train, X_test_df, y_test, final_selected_features)

if N_CLASSES > 2:
    plot_gs_metrics(train_probs, test_probs, y_train, y_test)
else:
    plot_binary_metrics(train_probs, test_probs, y_train, y_test)
# %%
