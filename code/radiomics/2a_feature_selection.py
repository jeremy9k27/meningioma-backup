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
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import get_data
from preprocessing.utils import setup
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

setup()

OUTCOME = 'MethylationSubgroup' # 'MethylationSubgroup' or 'Chr1p' or 'Chr22q' or 'Chr9p' or 'TERT'
CLASS_IDS = ['Merlin Intact', 'Immune Enriched', 'Hypermetabolic'] # ['Intact', 'Loss'] or ['Merlin Intact', 'Immune Enriched', 'Hypermetabolic']
N_CLASSES = len(CLASS_IDS)
TEST_SIZE = 12
SEED = 1
MULTIPLE_IMPUTATION = False # Kernel keeps crashing..! OOM? or try mice package later
USE_SMOTE = False
OUTPUT_DIR = f'data/radiomics/evaluations/{OUTCOME}_TestSize-{TEST_SIZE}_Seed-{SEED}_MultImpute-{MULTIPLE_IMPUTATION}_SMOTE-{USE_SMOTE}'
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def prep_data(outcome=OUTCOME, test_size=TEST_SIZE, seed=SEED, even_test_split=USE_SMOTE, multiple_imputation=MULTIPLE_IMPUTATION):
    train_df, test_df = get_data(outcome=outcome, test_size=test_size, seed=seed, even_test_split=even_test_split)
    if multiple_imputation:
        X_train_df = train_df.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
        y_train = train_df[outcome].values.astype(int)
        X_test_df = test_df.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
        y_test = test_df[outcome].values.astype(int)
        scaler = StandardScaler()
        X_train_df = pd.DataFrame(scaler.fit_transform(X_train_df), columns=X_train_df.columns)
        X_test_df = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns)
        imputer = IterativeImputer(max_iter=2, sample_posterior=False, random_state=seed, verbose=2)
        X_train_imputed_df = pd.DataFrame(imputer.fit_transform(X_train_df), columns=X_train_df.columns)
        X_test_imputed_df = pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns)
        X_train_df = X_train_imputed_df
        X_test_df = X_test_imputed_df
    else:
        train_df = train_df.fillna(train_df.mean())
        test_df = test_df.fillna(train_df.mean())
        X_train_df = train_df.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
        y_train = train_df[outcome].values.astype(int)
        X_test_df = test_df.drop(columns=['Subject Number', 'MethylationSubgroup', 'Chr1p', 'Chr22q', 'Chr9p', 'TERT'])
        y_test = test_df[outcome].values.astype(int)
    
    # drop columns in train_df and test_df that are all NaN
    X_train_df = X_train_df.dropna(axis=1, how='all')
    X_test_df = X_test_df[X_train_df.columns]

    if USE_SMOTE:
        scaler = MinMaxScaler()
        X_train_df = pd.DataFrame(scaler.fit_transform(X_train_df), columns=X_train_df.columns)
        X_test_df = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns)
    
    return X_train_df, y_train, X_test_df, y_test

def plot_train_test_split(y_train, y_test, save_fig=True):
    unique_classes = np.unique(np.concatenate((y_train, y_test)))
    train_counts_array = [np.sum(y_train == uc) for uc in unique_classes]
    test_counts_array = [np.sum(y_test == uc) for uc in unique_classes]
    combined_counts_array = pd.DataFrame({'Training': train_counts_array, 'Testing': test_counts_array}, index=unique_classes)
    
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(unique_classes))

    # Plotting the training bars
    train_bars = ax.bar(index - bar_width/2, combined_counts_array['Training'], bar_width, label='Training', color='tab:blue')

    # Plotting the testing bars
    test_bars = ax.bar(index + bar_width/2, combined_counts_array['Testing'], bar_width, label='Testing', color='tab:orange')

    # Adding the bar labels
    ax.bar_label(train_bars, padding=3)
    ax.bar_label(test_bars, padding=3)

    # Setting the rest of the plot
    ax.set_xlabel('Class')
    ax.set_xticks(index)
    ax.set_xticklabels(CLASS_IDS if CLASS_IDS else unique_classes)
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Train/Test Samples per Class ({len(y_train)}/{len(y_test)} split)')
    ax.legend(title='Dataset')

    plt.tight_layout()

    # Saving the figure or showing it
    if save_fig:
        plt.savefig(f'{OUTPUT_DIR}/train_test_split.png')
    else:
        plt.show()

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

def run_model(X_train_df, y_train, X_test_df, y_test, features, n_classes=N_CLASSES, seed=SEED, smote=USE_SMOTE):
    # Prep data for classifier
    X_train = X_train_df[features]
    if smote:
        print(f"Before SMOTE: X_train: {X_train.shape}, y_train: {y_train.shape}")
        smote = SMOTE(random_state=seed)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: X_train: {X_train.shape}, y_train: {y_train.shape}")

    X_test = X_test_df[features]
    # if n_classes > 2:
    #     y_train = label_binarize(y_train, classes=np.arange(n_classes))
    #     y_test = label_binarize(y_test, classes=np.arange(n_classes))

    # Train classifier
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    estimator = RandomForestClassifier()
    params = {
        'n_estimators': [75, 100, 150, 500, 1000],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 20, 50],
        'min_samples_split': [1, 2, 4],
        'bootstrap': [True, False]
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
    # rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=seed) # TODO: vary hyperparameters
    # rf_classifier.fit(X_train, y_train)

    # # Predicting probabilities for train/test set
    # train_probs = rf_classifier.predict_proba(X_train)
    # test_probs = rf_classifier.predict_proba(X_test)

    # return train_probs, test_probs, y_train, y_test

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

def plot_binary_metrics(train_probs, test_probs, y_train, y_test, use_test=True, class_ids=CLASS_IDS, save_fig=True):
    if not use_test:
        y_test = y_train
        test_probs = train_probs

    fpr, tpr, _ = roc_curve(y_test, test_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 2)
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

    predicted_labels = np.argmax(test_probs, axis=1)
    conf_matrix = confusion_matrix(y_test, predicted_labels)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = accuracy_score(y_test, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='viridis', cbar=False, xticklabels=class_ids, yticklabels=class_ids)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix: {OUTCOME} (Train/test split: {y_train.shape[0]}/{TEST_SIZE})\nOverall Accuracy = {accuracy*100:.2f}%')
    
    if save_fig:
        plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png')
    else:
        plt.show()

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

    # Plotting and saving the table as a PNG
    _, ax = plt.subplots(figsize=(8, 3))  # Adjust the figure size as needed
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc = 'center', loc='center')
    if save_fig:
        plt.savefig(f'{OUTPUT_DIR}/metrics_table.png')
    else:
        plt.show()

def plot_gs_metrics(train_probs, test_probs, y_train, y_test, use_test=True, class_ids=CLASS_IDS, save_fig=True):
    if not use_test:
        y_test = y_train
        test_probs = train_probs
    
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
    # save plot 1
    if save_fig: 
        plt.savefig(f'{OUTPUT_DIR}/roc_curve.png')
    else:
        plt.show()

    # create plot 2
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

#%%
X_train_df, y_train, X_test_df, y_test = prep_data()
plot_train_test_split(y_train, y_test)

final_selected_features, final_selected_feature_ranks = feature_selection(X_train_df, y_train)
plot_corr_matrix(X_train_df[final_selected_features])

# train_probs, test_probs, y_train, y_test = run_model(X_train_df, y_train, X_test_df, y_test, final_selected_features)
train_probs, test_probs, y_train, y_test, results_df, best_params, best_score, classifier = run_model(X_train_df, y_train, X_test_df, y_test, final_selected_features)

if N_CLASSES > 2:
    plot_gs_metrics(train_probs, test_probs, y_train, y_test)
else:
    plot_binary_metrics(train_probs, test_probs, y_train, y_test)
# %%
