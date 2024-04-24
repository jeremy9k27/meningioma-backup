# TODO:
# look thru results of each pairing for model with highest validation accuracy (pretend u dont have test set results)
# then ensemble the different kinds of models together, examine train and test set score of the ensemble

# still need to do nested approach instead of current approach..!
# still need to incorporate collage features
# also try active learning at this point
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import plot_train_test_split, plot_corr_matrix, prep_data_for_loocv
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, jaccard_score
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import joblib
import os
import sys
import time
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import cross_val_score

class LOOExperiment:
    def __init__(self, prediction_task, lambdas=[0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15], use_smote=True, scaler='Standard', seed=0, output_dir='data/radiomics_loo', save=True):
        """
        Initialize the experiment with the provided settings. 
        
        Notes
        -----
        * scaler: The scaler to use for the experiment. If None, no scaling will be applied. Can be None, 'Standard', or 'MinMax'.
        * use_smote: Whether to use SMOTE for oversampling the minority class. If True, the scaler must be provided. Only makes sense for imbalanced datasets (i.e., NOT MethylationSubgroup).
        """
        # User specified settings
        self.prediction_task = prediction_task
        self.lambdas = lambdas
        self.use_smote = use_smote
        self.scaler = scaler
        self.seed = seed
        self.output_dir = f"{output_dir}/{prediction_task}"
        self.classic_loo_output_dir = f"{self.output_dir}/classic_loo"
        self.classic_loo_evolution_dir = f"{self.classic_loo_output_dir}/evolution_curves"
        self.lto_output_dir = f"{self.output_dir}/lto"
        for d in [self.output_dir, self.classic_loo_output_dir, self.classic_loo_evolution_dir, self.lto_output_dir]:
            if not os.path.exists(d) and save: os.makedirs(d)
        
        self.save = save

        # Setting we don't typically need to change...
        self.feat_file = f"data/radiomics/features6/features_wide.csv" # File location with the radiomics features in wide format
        self.lr_params = {
            'penalty': 'l1', 
            'class_weight': 'balanced', 
            'solver': 'liblinear',
            'random_state': self.seed, 
            'max_iter':100, 
            'verbose':0
        }
        # Setting up the logger
        # self.log_dir = f"{self.output_dir}/logs"
        # if not os.path.exists(self.log_dir) and save: os.makedirs(self.log_dir)
        # self.logger = logging.getLogger(f'{self.output_dir}/Exp_PPID-{parallel_process_id}')
        # self.logger.setLevel(logging.INFO)
        # if not self.logger.handlers:
        #     if save:
        #         fh = logging.FileHandler(f"{self.log_dir}/Exp_PPID-{parallel_process_id}.log")
        #     else:
        #         fh = logging.StreamHandler(sys.stdout)
        #     formatter = logging.Formatter('%(asctime)s (%(levelname)s): %(message)s', datefmt='%m/%d %H:%M')
        #     fh.setFormatter(formatter)
        #     self.logger.addHandler(fh)

        # Automatically generated settings
        if self.prediction_task == 'MethylationSubgroup':
            self.class_ids = ['Merlin Intact', 'Immune Enriched', 'Hypermetabolic']
        else:
            self.class_ids = ['Intact', 'Loss']
        self.n_classes = len(self.class_ids)
        if self.use_smote: assert self.scaler is not None, "Scaler must be provided for SMOTE"
        if self.scaler == 'Standard':
            self.scaler_obj = StandardScaler()
        elif self.scaler == 'MinMax':
            self.scaler_obj = MinMaxScaler()
        else:
            self.scaler_obj = None

        # Reading in the data
        self.X, self.y = prep_data_for_loocv(
            features_file=self.feat_file, 
            outcome=self.prediction_task, 
            scaler_obj=self.scaler_obj
        )

        # Remove constant features
        self.constant_feats = [col for col in self.X.columns if self.X[col].nunique() == 1]
        self.X = self.X.drop(columns=self.constant_feats)
        self.feat_names = self.X.columns

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plots a confusion matrix given y_true labels and y_pred predictions and returns the matrix."""
        conf_matrix = confusion_matrix(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='viridis', cbar=False, xticklabels=self.class_ids, yticklabels=self.class_ids)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix (lambda = {self.best_lambda})\nBalanced Accuracy = {balanced_accuracy*100:.2f}%')
        
        if self.save:
            plt.savefig(f'{self.classic_loo_output_dir}/final_test_confusion_matrix.png')
        else:
            plt.show()
        plt.close()

        return conf_matrix, balanced_accuracy

    def _plot_metrics_table(self, metrics):
        """Plots / saves a table of performance metrics. It is expected that metrics is a dictionary with metric names as keys and metric values as values."""

        # Conver the dict to a df
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])

        # Make the metrics table
        _, ax = plt.subplots(figsize=(8, 3))
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc = 'center', loc='center')

        if self.save:
            # Save the metrics table
            plt.savefig(f'{self.classic_loo_output_dir}/final_test_perf_metrics.png')
            plt.close()
            
            # Save the metrics to their own csv file
            metrics_df.to_csv(f'{self.classic_loo_output_dir}/final_test_perf_metrics.csv', index=False)
        else:
            plt.show()
            plt.close()
    
    def get_binary_metrics(self, probs, y_pred, y_true):
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        metrics = {
            'AUC': roc_auc,
            'Binary F1': f1_score(y_true, y_pred, average='binary'),
            'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
            'Binary Precision': precision_score(y_true, y_pred, average='binary'),
            'Weighted Precision': precision_score(y_true, y_pred, average='weighted'),
            'Binary Recall (Sensitivity)': recall_score(y_true, y_pred, average='binary'),
            'Weighted Recall (Sensitivity)': recall_score(y_true, y_pred, average='weighted'),
            'Specificity': tn / (tn + fp),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Binary Jaccard': jaccard_score(y_true, y_pred, average='binary'),
            'Weighted Jaccard': jaccard_score(y_true, y_pred, average='weighted')
        }
        return metrics

    def get_multiclass_metrics(self, probs, y_pred, y_true):
        fpr, tpr, roc_auc = dict(), dict(), dict()

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr_grid = np.linspace(0.0, 1.0, 1000)

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(self.n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

        # Average it and compute AUC
        mean_tpr /= self.n_classes

        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        y_true = np.argmax(y_true, axis=1)

        metrics = {
            'Macro AUC': roc_auc["macro"],
            'Micro AUC': roc_auc["micro"],
            'Macro F1': f1_score(y_true, y_pred, average='macro'),
            'Micro F1': f1_score(y_true, y_pred, average='micro'),
            'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
            'Macro Precision': precision_score(y_true, y_pred, average='macro'),
            'Micro Precision': precision_score(y_true, y_pred, average='micro'),
            'Weighted Precision': precision_score(y_true, y_pred, average='weighted'),
            'Macro Recall (Sensitivity)': recall_score(y_true, y_pred, average='macro'),
            'Micro Recall (Sensitivity)': recall_score(y_true, y_pred, average='micro'),
            'Weighted Recall (Sensitivity)': recall_score(y_true, y_pred, average='weighted'),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Macro Jaccard': jaccard_score(y_true, y_pred, average='macro'),
            'Micro Jaccard': jaccard_score(y_true, y_pred, average='micro'),
            'Weighted Jaccard': jaccard_score(y_true, y_pred, average='weighted')
        }

        return metrics
        
    def _plot_binary_results(self, probs, y_true):
        """Plot ROC curve, confusion matrix, and metrics table for binary classification tasks. Returns the ROC AUC score."""

        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plot 1/3: ROC curve:
        plt.plot(fpr, tpr, color='tab:orange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='tab:blue', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (lambda = {self.best_lambda})')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{self.classic_loo_output_dir}/final_test_roc_curve.png')
        else:
            plt.show()
        plt.close()

        # Plot 2/3: Confusion matrix
        y_pred = np.argmax(probs, axis=1)
        conf_matrix, balanced_accuracy = self._plot_confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = conf_matrix.ravel()

        # Plot 3/3: Metrics table
        metrics = {
            'AUC': roc_auc,
            'Binary F1': f1_score(y_true, y_pred, average='binary'),
            'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
            'Binary Precision': precision_score(y_true, y_pred, average='binary'),
            'Weighted Precision': precision_score(y_true, y_pred, average='weighted'),
            'Binary Recall (Sensitivity)': recall_score(y_true, y_pred, average='binary'),
            'Weighted Recall (Sensitivity)': recall_score(y_true, y_pred, average='weighted'),
            'Specificity': tn / (tn + fp),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy,
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Binary Jaccard': jaccard_score(y_true, y_pred, average='binary'),
            'Weighted Jaccard': jaccard_score(y_true, y_pred, average='weighted')
        }

        self._plot_metrics_table(metrics)

    def _plot_multiclass_results(self, probs, y_true):
        """Plot ROC curve, confusion matrix, and metrics table for multiclass classification tasks. Expects y_true to be one-hot encoded."""

        fpr, tpr, roc_auc = dict(), dict(), dict()

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr_grid = np.linspace(0.0, 1.0, 1000)

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(self.n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

        # Average it and compute AUC
        mean_tpr /= self.n_classes

        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot 1/3: ROC curve
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
        for i, color, class_id in zip(range(self.n_classes), colors, self.class_ids):
            RocCurveDisplay.from_predictions(
                y_true[:, i],
                probs[:, i],
                name=f"ROC curve for {class_id}",
                color=color,
                ax=ax,
                plot_chance_level=(i == self.n_classes - 1),
            )

        _ = ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"{self.prediction_task}: One-vs-Rest ROC Curves (lambda = {self.best_lambda})",
        )

        if self.save: 
            plt.savefig(f'{self.classic_loo_output_dir}/final_test_roc_curve.png')
        else:
            plt.show()

        plt.close()

        # Plot 2/3: Confusion matrix
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(probs, axis=1)
        self._plot_confusion_matrix(y_true, y_pred)

        # Plot 3/3: Metrics table
        metrics = {
            'Macro AUC': roc_auc["macro"],
            'Micro AUC': roc_auc["micro"],
            'Macro F1': f1_score(y_true, y_pred, average='macro'),
            'Micro F1': f1_score(y_true, y_pred, average='micro'),
            'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
            'Macro Precision': precision_score(y_true, y_pred, average='macro'),
            'Micro Precision': precision_score(y_true, y_pred, average='micro'),
            'Weighted Precision': precision_score(y_true, y_pred, average='weighted'),
            'Macro Recall (Sensitivity)': recall_score(y_true, y_pred, average='macro'),
            'Micro Recall (Sensitivity)': recall_score(y_true, y_pred, average='micro'),
            'Weighted Recall (Sensitivity)': recall_score(y_true, y_pred, average='weighted'),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Macro Jaccard': jaccard_score(y_true, y_pred, average='macro'),
            'Micro Jaccard': jaccard_score(y_true, y_pred, average='micro'),
            'Weighted Jaccard': jaccard_score(y_true, y_pred, average='weighted')
        }

        self._plot_metrics_table(metrics)

        return metrics['Macro F1']

    def lr(self, coef, intercept, query):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        raw = sigmoid(np.squeeze(np.dot(coef, query.T)) + intercept)
        if len(raw) == 1:
            raw = np.array([1 - raw[0], raw[0]])

        probs = raw / np.sum(raw)
        pred = int(np.argmax(probs))

        return probs, pred
    
    def plot_metric_by_lambda(self, train_metrics_by_lambda, test_metrics_by_lambda, metric):
        # Processing Training Data
        train_means = {}
        train_stds = {}
        for key, df in train_metrics_by_lambda.items():
            if metric in df.columns:
                train_means[key] = df[metric].mean()
                train_stds[key] = df[metric].std()
            else:
                print(f"Metric {metric} not found in training data for lambda {key}")

        # Converting dictionaries to DataFrames
        train_mean_df = pd.DataFrame(list(train_means.items()), columns=['Lambda', 'Mean'])
        train_std_df = pd.DataFrame(list(train_stds.items()), columns=['Lambda', 'Std Dev'])

        # Processing Test Data
        test_metrics = {k: df[metric].iloc[0] for k, df in test_metrics_by_lambda.items() if metric in df.columns}
        test_df = pd.DataFrame(list(test_metrics.items()), columns=['Lambda', metric])

        # Plotting Training Data
        plt.errorbar(train_mean_df['Lambda'], train_mean_df['Mean'], yerr=train_std_df['Std Dev'], label=f'Train {metric}', marker='o')

        # Plotting Test Data
        plt.plot(test_df['Lambda'], test_df[metric], label=f'Test {metric}', linestyle='--', marker='x')

        plt.xlabel('Lambda')
        plt.ylabel(metric)
        plt.title(f'{self.prediction_task}: Classic LOO CV Evolution Curve')
        plt.legend()
        plt.grid(True)
        if self.save:
            plt.savefig(f'{self.classic_loo_evolution_dir}/{metric}.png')
        else:
            plt.show()
        plt.close()

        return test_df

    def plot_heatmap(self, data, name, plot_name=None):
        plt.figure(figsize=(max(0.9*data.shape[0], 14), data.shape[0]))  # Adjust the figure size as needed
        sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Scale'})
        plt.xlabel('Fold', fontsize=16)
        plt.ylabel('Feature', fontsize=16)
        plt.yticks(rotation=0, fontsize=12)
        plt.title(f'{name} Heatmap of L1-regularized coefs across folds\n(lambda = {self.best_lambda})', fontsize=20)
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{self.classic_loo_output_dir}/{plot_name}_coefs_heatmap.png')
        else:
            plt.show()
        plt.close()

    def plot_coef_bargraph(self, data, name, plot_name=None):
        plt.figure(figsize=(12, 12))
        data = data.abs()
        data = data.iloc[::-1]
        data.plot(kind='barh', stacked=True, colormap='viridis', ax=plt.gca(), legend=False)
        plt.title(f'{name} Sum of Coef Absolute Values \n(lambda = {self.best_lambda})', fontsize=20)
        plt.ylabel('Feature', fontsize=16)
        plt.xlabel('Sum of Coef Absolute Values (coloured by fold #)', fontsize=16)
        plt.yticks(rotation=45, fontsize=12)
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{self.classic_loo_output_dir}/{plot_name}_coefs_bargraph.png')
        else:
            plt.show()
        plt.close()
    
    def plot_var_exp(self, data, name, plot_name=None, hline=False):
        norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())

        # Create a ScalarMappable and initialize a colormap
        colormap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 12))
        data.plot(kind='bar', color=[colormap(norm(value)) for value in data], ax=ax)

        # Optionally add a colorbar
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Variance Explained')

        fig.suptitle(name, fontsize=20)

        if hline:        
            ax.axhline(y=0.99, color='black', linestyle='--', linewidth=1)
            ax.text(x=-0.5, y=0.99, s='y = 0.99', verticalalignment='bottom', color='black')

        plt.tight_layout()

        # Show the plot
        if self.save:
            plt.savefig(f'{self.classic_loo_output_dir}/{plot_name}.png')
        else:
            plt.show()
        plt.close()

    def plot_coefs(self):
        if self.n_classes > 2:
            for k, biomarker in enumerate(self.class_ids):
                coefs = pd.DataFrame(self.coef[:, k, :], columns=self.feat_names)
                coefs['intercept'] = self.intercept[:, k]
                zero_feats = [col for col in coefs.columns if (coefs[col] == 0.).all()]
                nonzero_coefs = coefs.drop(columns=zero_feats)
                nonzero_coefs = nonzero_coefs.T

                nonzero_coefs['Absolute Sum'] = nonzero_coefs.abs().sum(axis=1)
                nonzero_coefs = nonzero_coefs.sort_values(by='Absolute Sum', ascending=False)

                nonzero_coefs['Prop Var Exp'] = nonzero_coefs['Absolute Sum'] / nonzero_coefs['Absolute Sum'].sum()
                self.plot_var_exp(nonzero_coefs['Prop Var Exp'], f"{biomarker} vs. Rest: Proportion of Variance Explained", plot_name=f"{biomarker}_prop_var_exp.png")
                
                nonzero_coefs['Cum Var Exp'] = nonzero_coefs['Prop Var Exp'].cumsum()
                self.plot_var_exp(nonzero_coefs['Cum Var Exp'], f"{biomarker} vs. Rest: Cumulative Variance Explained", plot_name=f"{biomarker}_cum_var_exp.png", hline=True)
                
                nonzero_coefs_filtered = nonzero_coefs[nonzero_coefs['Cum Var Exp'] < 0.99]
                nonzero_coefs_final = nonzero_coefs_filtered.drop(columns=['Absolute Sum', 'Prop Var Exp', 'Cum Var Exp'])

                self.plot_coef_bargraph(nonzero_coefs_final, f"{biomarker} vs. Rest:", plot_name=biomarker)
                self.plot_heatmap(nonzero_coefs_final, f"{biomarker} vs. Rest:", plot_name=biomarker)

                if self.save: nonzero_coefs.to_csv(f'{self.classic_loo_output_dir}/{biomarker}_coefs.csv')

        else:
            coefs = pd.DataFrame(self.coef.squeeze(), columns=self.feat_names)
            coefs['intercept'] = self.intercept
            zero_feats = [col for col in coefs.columns if (coefs[col] == 0.).all()]
            nonzero_coefs = coefs.drop(columns=zero_feats)
            nonzero_coefs = nonzero_coefs.T

            nonzero_coefs['Absolute Sum'] = nonzero_coefs.abs().sum(axis=1)
            nonzero_coefs = nonzero_coefs.sort_values(by='Absolute Sum', ascending=False)

            nonzero_coefs['Prop Var Exp'] = nonzero_coefs['Absolute Sum'] / nonzero_coefs['Absolute Sum'].sum()
            self.plot_var_exp(nonzero_coefs['Prop Var Exp'], f"{self.prediction_task}: Proportion of Variance Explained", plot_name="prop_var_exp.png")

            nonzero_coefs['Cum Var Exp'] = nonzero_coefs['Prop Var Exp'].cumsum()
            self.plot_var_exp(nonzero_coefs['Cum Var Exp'], f"{self.prediction_task}: Cumulative Variance Explained", plot_name="cum_var_exp.png", hline=True)

            nonzero_coefs_filtered = nonzero_coefs[nonzero_coefs['Cum Var Exp'] < 0.99]
            nonzero_coefs_final = nonzero_coefs_filtered.drop(columns=['Absolute Sum', 'Prop Var Exp', 'Cum Var Exp'])

            self.plot_coef_bargraph(nonzero_coefs_final, self.prediction_task, plot_name=self.prediction_task)
            self.plot_heatmap(nonzero_coefs_final, self.prediction_task, plot_name=self.prediction_task)

            if self.save: nonzero_coefs.to_csv(f'{self.classic_loo_output_dir}/{self.prediction_task}_coefs.csv')
        
        return nonzero_coefs

    def loo_model(self, pmetric='AUC'):
        
        coefs_by_lambda = {}
        intercepts_by_lambda = {}
        test_probs_by_lambda = {}
        train_metrics_by_lambda = {}
        test_metrics_by_lambda = {}

        for i, lmda in enumerate(self.lambdas):
            
            coefs = []
            intercepts = []
            test_probs = []
            test_preds = []
            train_metrics = []

            for train_idx, test_idx in tqdm(LeaveOneOut().split(self.X), total=len(self.X), desc=f"Processing lambda ({i+1}/{len(self.lambdas)})"):

                X_train = self.X.iloc[train_idx]
                y_train = self.y[train_idx]
                X_test = self.X.iloc[test_idx]
                y_test = self.y[test_idx]

                train_idx_end = len(X_train)

                if self.use_smote:
                    X_train, y_train = SMOTE(random_state=self.seed).fit_resample(X_train, y_train)

                model = LogisticRegression(C=lmda, **self.lr_params)

                model.fit(X_train, y_train)

                coefs.append(model.coef_)
                intercepts.append(model.intercept_)

                train_probs = model.predict_proba(X_train.iloc[0:train_idx_end])
                train_preds = model.predict(X_train.iloc[0:train_idx_end])

                if self.n_classes == 2:
                    train_metrics.append(self.get_binary_metrics(train_probs, train_preds, y_train[0:train_idx_end]))
                else:
                    train_metrics.append(self.get_multiclass_metrics(train_probs, train_preds, label_binarize(y_train[0:train_idx_end], classes=np.arange(self.n_classes))))

                test_probs.append(model.predict_proba(X_test))
                test_preds.append(model.predict(X_test))
            
            # Save results for this lambda here
            test_preds = np.squeeze(np.stack(test_preds))
            test_probs = np.squeeze(np.stack(test_probs))
            test_probs_by_lambda[lmda] = test_probs
            coefs_by_lambda[lmda] = np.squeeze(np.stack(coefs))
            intercepts_by_lambda[lmda] = np.squeeze(np.stack(intercepts))

            train_metrics_by_lambda[lmda] = pd.DataFrame(train_metrics)
            if self.n_classes == 2:
                test_metrics_by_lambda[lmda] = pd.DataFrame(self.get_binary_metrics(test_probs, test_preds, self.y), index=[0])
            else:
                test_metrics_by_lambda[lmda] = pd.DataFrame(self.get_multiclass_metrics(test_probs, test_preds, label_binarize(self.y, classes=np.arange(self.n_classes))), index=[0])

        self.perf_metric = pmetric
        test_perf_met_by_lambda = self.plot_metric_by_lambda(train_metrics_by_lambda, test_metrics_by_lambda, pmetric)
        for metric in test_metrics_by_lambda[self.lambdas[0]].columns:
            self.plot_metric_by_lambda(train_metrics_by_lambda, test_metrics_by_lambda, metric)

        max_perf_met_index = test_perf_met_by_lambda[pmetric].idxmax()
        self.best_lambda = test_perf_met_by_lambda.loc[max_perf_met_index, 'Lambda']

        print(f"Best lambda: {self.best_lambda}")

        self.coef = coefs_by_lambda[self.best_lambda]
        self.intercept = intercepts_by_lambda[self.best_lambda]
        self.test_probs = test_probs_by_lambda[self.best_lambda]

        self.nonzero_coefs = self.plot_coefs()
        if self.n_classes > 2:
            self._plot_multiclass_results(self.test_probs, label_binarize(self.y, classes=np.arange(self.n_classes)))
        else:
            self._plot_binary_results(self.test_probs, self.y)

        return train_metrics_by_lambda, test_metrics_by_lambda, self.nonzero_coefs, self.best_lambda

    # Helper function to perform the computation for a single lambda
    def model_fit_predict(self, train_idx, test_idx, lmda):
        X_train = self.X.iloc[train_idx]
        y_train = self.y[train_idx]
        X_test = self.X.iloc[test_idx]
        y_test = self.y[test_idx]

        if self.use_smote:
            X_train, y_train = SMOTE(random_state=self.seed).fit_resample(X_train, y_train)

        model = LogisticRegression(C=lmda, **self.lr_params)
        model.fit(X_train, y_train)

        coefs = model.coef_
        intercepts = model.intercept_
        train_probs = model.predict_proba(X_train)
        train_preds = model.predict(X_train)

        test_probs = model.predict_proba(X_test)
        test_preds = model.predict(X_test)

        if self.n_classes == 2:
            train_metrics = self.get_binary_metrics(train_probs, train_preds, y_train)
        else:
            train_metrics = self.get_multiclass_metrics(train_probs, train_preds, label_binarize(y_train, classes=np.arange(self.n_classes)))

        return (coefs, intercepts, test_probs, test_preds, y_test, train_metrics)

    def process_lambda(self, lmda):
        results = []

        # Create a ProcessPoolExecutor to handle tasks in parallel
        with ProcessPoolExecutor(max_workers=16) as executor:
            # Create tasks for each train-test split
            futures = [executor.submit(self.model_fit_predict, train_idx, test_idx, lmda)
                       for train_idx, test_idx in LeaveOneOut().split(self.X)]

            # Use tqdm to monitor progress
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing lambda={lmda}"):
                results.append(future.result())

        # Unpack results
        coefs, intercepts, test_probs, test_preds, y_test, train_metrics = zip(*results)

        return {
            'lambda': lmda,
            'coefs': np.stack(coefs).squeeze(),
            'intercepts': np.stack(intercepts).squeeze(),
            'test_probs': np.stack(test_probs).squeeze(),
            'test_preds': np.stack(test_preds).squeeze(),
            'y_test': np.stack(y_test).squeeze(),
            'train_metrics': pd.DataFrame(list(train_metrics))
        }

    def par_loo_model(self, pmetric='AUC'):
        # Preparing storage dictionaries
        coefs_by_lambda = {}
        intercepts_by_lambda = {}
        test_probs_by_lambda = {}
        y_tests_by_lambda = {}
        train_metrics_by_lambda = {}
        test_metrics_by_lambda = {}
        
        # Use ProcessPoolExecutor to parallelize the loop over lambdas
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(self.process_lambda, lmda): lmda for lmda in self.lambdas}
            for future in as_completed(futures):
                result = future.result()
                lmda = result['lambda']
                coefs_by_lambda[lmda] = result['coefs']
                intercepts_by_lambda[lmda] = result['intercepts']
                test_probs_by_lambda[lmda] = result['test_probs']
                test_preds = result['test_preds']
                y_tests_by_lambda[lmda] = result['y_test']
                train_metrics_by_lambda[lmda] = result['train_metrics']
                if self.n_classes == 2:
                    test_metrics = pd.DataFrame(self.get_binary_metrics(result['test_probs'], test_preds, result['y_test']), index=[0])
                else:
                    test_metrics = pd.DataFrame(self.get_multiclass_metrics(result['test_probs'], test_preds, label_binarize(result['y_test'], classes=np.arange(self.n_classes))), index=[0])

                test_metrics_by_lambda[lmda] = test_metrics
        
        self.perf_metric = pmetric
        test_perf_met_by_lambda = self.plot_metric_by_lambda(train_metrics_by_lambda, test_metrics_by_lambda, pmetric)
        for metric in test_metrics_by_lambda[self.lambdas[0]].columns:
            self.plot_metric_by_lambda(train_metrics_by_lambda, test_metrics_by_lambda, metric)

        max_perf_met_index = test_perf_met_by_lambda[pmetric].idxmax()
        self.best_lambda = test_perf_met_by_lambda.loc[max_perf_met_index, 'Lambda']

        print(f"Best lambda: {self.best_lambda}")

        self.coef = coefs_by_lambda[self.best_lambda]
        self.intercept = intercepts_by_lambda[self.best_lambda]
        self.test_probs = test_probs_by_lambda[self.best_lambda]
        self.y_test = y_tests_by_lambda[self.best_lambda]

        self.nonzero_coefs = self.plot_coefs()
        if self.n_classes > 2:
            self._plot_multiclass_results(self.test_probs, label_binarize(self.y_test, classes=np.arange(self.n_classes)))
        else:
            self._plot_binary_results(self.test_probs, self.y_test)

        self.train_metrics_by_lambda = train_metrics_by_lambda
        self.test_metrics_by_lambda = test_metrics_by_lambda

        return self.train_metrics_by_lambda, self.test_metrics_by_lambda, self.nonzero_coefs, self.best_lambda

    # def inner_loop(X, y, lambda):
    #     """
    #     Dataset (X, y) is assumed to have N many samples, K many features. Lambda is fixed
    #     cross val returns N many predictions, one for each sample in X, and N many models, one fit on each of the N possible training sets
    #     """
    #     N_preds, N_models = cross_val(LogReg(X, y, lambda), splits=LOO(X, y))
    #     metric = balanced_accuracy_score(y, N_preds)
    #     return metric, N_models
    #
    # N = len(X)
    # for lambda in [0.1, 1, 10, ...]:
    #     all_y_hats = []
    #     val_metrics = []
    #     for i in range(N):
    #
    #         X_train = X.drop(i)
    #         y_train = y.drop(i)
    #         X_test = X[i]
    #         y_test = y[i]
    #
    #         val_metric, N_models = inner_loop(X_train, y_train, lambda)
    #
    #         val_metrics.append(val_metric)
    #
    #         y_hat = predict(N_models, X_test)
    #         all_y_hats.append(y_hat)
    #
    #     test_metric = balanced_accuracy_score(y, all_y_hats)
    # each lambda is then left with: 1 test_metric, N val_metrics, and N*(N-1) train metrics
    # can plot a graph where X axis is lambda, Y axis is metric, and three are 3 lines, one for test performance, one for train, one for val.
    # error bars exist for train and val, which are the std of each metric respectively
        
    def debug(self, lambdas=[0.1]):
        """
        lambdas : list
            List of L1 regularization parameters to search over. [0.1, 1, 10, 100, 1000] ?
        """
        # Read in data
        X, y, subject_nums = prep_data_for_loocv(
            features_file=self.feat_file, 
            outcome=self.prediction_task, 
            scaler_obj=self.scaler_obj
        )

        # Remove constant features
        constant_feats = [col for col in X.columns if X[col].nunique() == 1]
        X = X.drop(columns=constant_feats)

        self.train_subjects_df = X.copy()
        # Plot class split
        plot_train_test_split(y, y, output_file=None, class_ids=self.class_ids)
    
        outer_loo = LeaveOneOut()
        outer_coefs = []
        outer_intercepts = []
        outer_preds = []
        outer_labels = []

        for lmda in lambdas:
            for i, (train_val_idx, test_idx) in enumerate(outer_loo.split(X)):
                print(f"Outer fold {i + 1}/{len(X)}")

                X_train_val = X.iloc[train_val_idx]
                y_train_val = y[train_val_idx]
                X_test = X.iloc[test_idx]
                y_test = y[test_idx]

                if self.use_smote:
                    X_train_val, y_train_val = SMOTE(random_state=self.seed).fit_resample(X_train_val, y_train_val)

                inner_loo = LeaveOneOut()

                inner_coefs = []
                inner_intercepts = []
                inner_preds = []
                inner_labels = []

                for j, (train_idx, val_idx) in enumerate(inner_loo.split(X_train_val)):
                    # print(f"Inner fold {j + 1}/{len(X_train_val)}")

                    X_train = X_train_val.iloc[train_idx]
                    y_train = y_train_val[train_idx]
                    X_val = X_train_val.iloc[val_idx]
                    y_val = y_train_val[val_idx]

                    # Define model
                    model = LogisticRegression(
                        penalty='l1', 
                        C=lmda, 
                        class_weight='balanced', 
                        random_state=self.seed, 
                        solver='liblinear', 
                        max_iter=1000, 
                        verbose=1
                    )

                    # Fit model
                    model.fit(X_train, y_train)

                    # Save coefs and intercepts
                    inner_coefs.append(model.coef_)
                    inner_intercepts.append(model.intercept_)

                    # Predict
                    inner_preds.append(model.predict(X_val))
                    inner_labels.append(y_val)
                
                inner_preds = np.array(inner_preds).squeeze()
                inner_labels = np.array(inner_labels).squeeze()
                inner_coefs = np.stack(inner_coefs)
                inner_intercepts = np.stack(inner_intercepts)
                # self._plot_confusion_matrix(y_true=inner_labels[:len(X)-1], y_pred=inner_preds[:len(X)-1]) # excluding synthetic features made by SMOTE

                outer_coef = np.mean(inner_coefs, axis=0)
                outer_intercept = np.mean(inner_intercepts, axis=0)
                outer_coefs.append(outer_coef)
                outer_intercepts.append(outer_intercept)

                outer_prob, outer_pred = self.lr(outer_coef, outer_intercept, X_test)
                outer_preds.append(outer_pred)
                outer_labels.append(y_test)

            return outer_coefs, outer_intercepts, outer_preds, outer_labels, X.columns
    
    def process_fold(self, args):
        i, (train_val_idx, test_idx), X, y, lmda = args
        print(f"Processing Outer fold {i + 1}")

        X_train_val = X.iloc[train_val_idx]
        y_train_val = y[train_val_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        if self.use_smote:
            X_train_val, y_train_val = SMOTE(random_state=self.seed).fit_resample(X_train_val, y_train_val)

        inner_loo = LeaveOneOut()
        inner_coefs = []
        inner_intercepts = []
        inner_preds = []
        inner_labels = []

        for j, (train_idx, val_idx) in enumerate(inner_loo.split(X_train_val)):
            X_train = X_train_val.iloc[train_idx]
            y_train = y_train_val[train_idx]
            X_val = X_train_val.iloc[val_idx]
            y_val = y_train_val[val_idx]

            model = LogisticRegression(
                penalty='l1', 
                C=lmda, 
                class_weight='balanced', 
                random_state=self.seed, 
                solver='liblinear', 
                max_iter=1000, 
                verbose=0  # Reduced verbosity for parallel run
            )
            model.fit(X_train, y_train)
            inner_coefs.append(model.coef_)
            inner_intercepts.append(model.intercept_)
            inner_preds.append(model.predict(X_val))
            inner_labels.append(y_val)

        inner_coefs = np.stack(inner_coefs)
        inner_intercepts = np.stack(inner_intercepts)
        
        outer_coef = np.mean(inner_coefs, axis=0)
        outer_intercept = np.mean(inner_intercepts, axis=0)

        # Example: Implement prediction based on outer_coef and outer_intercept
        outer_pred = model.predict(X_test)  # Simple example
        outer_label = y_test
        
        return outer_coef, outer_intercept, outer_pred, outer_label

    def perform_cross_validation(self, lambdas=[0.1]):
        # Read in data
        X, y, subject_nums = prep_data_for_loocv(
            features_file=self.feat_file, 
            outcome=self.prediction_task, 
            scaler_obj=self.scaler_obj
        )

        # Remove constant features
        constant_feats = [col for col in X.columns if X[col].nunique() == 1]
        X = X.drop(columns=constant_feats)

        self.train_subjects_df = X.copy()

        # Plot class split
        plot_train_test_split(y, y, output_file=None, class_ids=self.class_ids)

        outer_loo = LeaveOneOut()

        args = [(i, indices, X, y, lambdas[0]) for i, indices in enumerate(outer_loo.split(X))]

        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            # Submit all tasks and store the future instances
            futures = [executor.submit(self.process_fold, arg) for arg in args]

            results = []
            # Setup tqdm to monitor progress based on completion of futures
            for future in tqdm(as_completed(futures), total=len(futures), desc="Outer folds"):
                result = future.result()
                results.append(result)

        outer_coefs, outer_intercepts, outer_preds, outer_labels = zip(*results)
        return outer_coefs, outer_intercepts, outer_preds, outer_labels, X.columns
