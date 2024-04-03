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
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import get_data, plot_corr_matrix
from preprocessing.utils import setup
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import joblib

setup()

# Researcher specified global constants
FEAT_FILE = 'data/radiomics/features5/features_wide.csv'
OUTCOME = 'MethylationSubgroup' # 'MethylationSubgroup' or 'Chr1p' or 'Chr22q' or 'Chr9p' or 'TERT'
TEST_SIZE = 21
SEED = 7
USE_SMOTE = False
SCALER = None # 'Standard' or 'MinMax' or None
EVEN_TEST_SPLIT = False
OUTPUT_DIR = f'data/radiomics/evaluations5/{OUTCOME}_TestSize-{TEST_SIZE}_Seed-{SEED}_SMOTE-{USE_SMOTE}_EvenTestSplit-{EVEN_TEST_SPLIT}_Scaler-{SCALER}'

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
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def feature_selection(X, y, top_k=32):
    # Step 1: Univariate feature selection
    print("Step 1/3: Performing univariate feature selection...")
    selector = SelectKBest(mutual_info_classif, k=1024) # TODO: vary score function and k ?
    X_reduced = selector.fit_transform(X, y)
    univariate_selected_features = X.columns[selector.get_support()]

    # Step 2: Gridsearch for a RF classifier to use during recursive feature elimination in next step
    print("Step 2/3: Performing gridsearch for RF classifier to use during RFE step...")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    estimator = RandomForestClassifier()
    params = {
        'n_estimators': [25, 50, 75, 100, 200],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring='f1_macro',
        refit=False,
        n_jobs=4,
        cv=cv,
        return_train_score=True,
        verbose=10
    )

    gs.fit(X_reduced, y)

    joblib.dump(gs, f'{OUTPUT_DIR}/gridsearch1_feature_selection.joblib')

    print("\tBest parameters from gridsearch: ", gs.best_params_)
    print("\tBest score from gridsearch: ", gs.best_score_)

    # Step 3: Cross-validation and recursive feature elimination
    print("Step 3/3: Performing cross-validation and recursive feature elimination...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    feature_ranks = []
    for train_index, test_index in tqdm(kf.split(X_reduced), total=kf.get_n_splits()):
        X_train, X_test = X_reduced[train_index], X_reduced[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rf = RandomForestClassifier(**gs.best_params_, random_state=SEED) # TODO: vary hyperparameters
        rfe = RFE(estimator=rf, n_features_to_select=1, step=32, verbose=0) # TODO: vary step?
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

    # Step 4: Plot correlation matrix of final selected features if specified
    plot_corr_matrix(X[final_selected_features], outcome=OUTCOME, test_size=TEST_SIZE, output_dir=OUTPUT_DIR)

    return final_selected_features, final_selected_feature_ranks

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
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample']
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

final_selected_features, final_selected_feature_ranks = feature_selection(X_train_df, y_train)

train_probs, test_probs, y_train, y_test, results_df, best_params, best_score, classifier = run_model(X_train_df, y_train, X_test_df, y_test, final_selected_features)

if N_CLASSES > 2:
    plot_gs_metrics(train_probs, test_probs, y_train, y_test)
else:
    plot_binary_metrics(train_probs, test_probs, y_train, y_test)
# %%
