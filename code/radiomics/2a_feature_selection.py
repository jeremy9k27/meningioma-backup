# File: 2a_feature_selection.py
# Date: 03/16/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description:

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to
#
# This script relies on the following file(s) as inputs:
#   *
#   *
#
# This script generates the following file(s) as outputs:
#   *
#   *
#
# Warnings:

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import get_data
from preprocessing.utils import setup
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, confusion_matrix, accuracy_score
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

setup()

OUTCOME = 'MethylationSubgroup'
CLASS_IDS = ['Merlin Intact', 'Immune Enriched', 'Hypermetabolic']
N_CLASSES = len(CLASS_IDS)
TEST_SIZE = 9
SEED = 1
OUTPUT_DIR = f'data/radiomics/evaluations/{OUTCOME}_TestSize-{TEST_SIZE}_Seed-{SEED}'
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def prep_data(outcome=OUTCOME, test_size=TEST_SIZE, seed=SEED):
    train_df, test_df = get_data(outcome=outcome, test_size=test_size, seed=seed)
    train_df = train_df.fillna(train_df.mean())
    test_df = test_df.fillna(test_df.mean())
    # TODO: use multiple imputation to handle missing values, but be careful about data leakage
    X_train_df = train_df.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
    y_train = train_df[outcome].values.astype(int)
    X_test_df = test_df.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
    y_test = test_df[outcome].values.astype(int)
    return X_train_df, y_train, X_test_df, y_test

def feature_selection(X, y, seed=SEED, top_k=32):
    # Step 1: Univariate feature selection
    selector = SelectKBest(mutual_info_classif, k=512) # TODO: vary score function and k ?
    X_reduced = selector.fit_transform(X, y)
    univariate_selected_features = X.columns[selector.get_support()]

    # Step 2: Cross-validation and recursive feature elimination
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    feature_ranks = []
    for train_index, test_index in tqdm(kf.split(X_reduced), total=kf.get_n_splits()):
        X_train, X_test = X_reduced[train_index], X_reduced[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rf = RandomForestClassifier(n_estimators=1000, random_state=seed) # TODO: vary hyperparameters
        rfe = RFE(estimator=rf, n_features_to_select=1, step=16, verbose=0) # TODO: vary step?
        rfe.fit(X_train, y_train)
        
        # TODO: check on performance metrics of rfe?

        # Rank features
        feature_ranks.append(rfe.ranking_)

    feature_ranks = np.stack(feature_ranks, axis=1)
    average_ranks = np.mean(feature_ranks, axis=1)

    # Step 3: Select the 32 best features based on average rank
    best_features_indices = np.argsort(average_ranks)[:top_k]
    final_selected_features = univariate_selected_features[best_features_indices]
    final_selected_feature_ranks = feature_ranks[best_features_indices, :]

    return final_selected_features, final_selected_feature_ranks

def clean_feature_names(strings):
    replacements = [
        "Mod-AX_3D_", "Mod-SAG_3D_", "_POST", "Mod-AX_", "SegLab-", "Feat-original_"
    ]
    
    # Replace specified substrings with ""
    replaced_strings = []
    for string in strings:
        for r in replacements:
            string = string.replace(r, "")
        # Replace "DIFFUSION" with "DWI"
        string = string.replace("DIFFUSION", "DWI")
        replaced_strings.append(string)
        
    return replaced_strings

def plot_corr_matrix(X, save_fig=True):
    normalizer = MinMaxScaler()
    X_normalized = pd.DataFrame(normalizer.fit_transform(X), columns=X.columns)
    corr_matrix = X_normalized.corr()
    clean_names = clean_feature_names(X.columns)
    plt.figure(figsize=(24, 20))
    sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="viridis", xticklabels=clean_names, yticklabels=clean_names)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='top')
    plt.title(f"Top Radiomics Features Correlation Matrix: {OUTCOME} (Train/test split: {X.shape[0]}/{TEST_SIZE})")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png')
    else:
        plt.show()

def run_model(X_train_df, y_train, X_test_df, y_test, features, n_classes=N_CLASSES, seed=SEED):
    # Prep data for classifier
    X_train = X_train_df[features]
    X_test = X_test_df[features]
    if n_classes > 2:
        y_train = label_binarize(y_train, classes=np.arange(n_classes))
        y_test = label_binarize(y_test, classes=np.arange(n_classes))

    # Train classifier
    # TODO: use cross-validation to tune hyperparameters
    rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=seed) # TODO: vary hyperparameters
    rf_classifier.fit(X_train, y_train)

    # Predicting probabilities for train/test set
    train_probs = rf_classifier.predict_proba(X_train)
    test_probs = rf_classifier.predict_proba(X_test)

    return train_probs, test_probs, y_train, y_test

def plot_metrics(train_probs, test_probs, y_train, y_test, use_test=True, class_ids=CLASS_IDS, save_fig=True):
    if not use_test:
        y_test = y_train
        test_probs = train_probs
    
    fpr, tpr, roc_auc = dict(), dict(), dict()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), np.array(test_probs)[:, :, 1].T.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], test_probs[i][:, 1])
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

    # create plot 1
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
            test_probs[i][:, 1],
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
    # save plot 1
    if save_fig: 
        plt.savefig(f'{OUTPUT_DIR}/roc_curve.png')
    else:
        plt.show()

    # create plot 2
    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(np.array(test_probs)[:, :, 1], axis=0)
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

#%%
X_train_df, y_train, X_test_df, y_test = prep_data()
final_selected_features, final_selected_feature_ranks = feature_selection(X_train_df, y_train)
print('Selected features:\n', final_selected_features)
print('Selected feature ranks:\n', final_selected_feature_ranks)
plot_corr_matrix(X_train_df[final_selected_features])
train_probs, test_probs, y_train, y_test = run_model(X_train_df, y_train, X_test_df, y_test, final_selected_features)
plot_metrics(train_probs, test_probs, y_train, y_test)