from sklearn.metrics import RocCurveDisplay, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, jaccard_score
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from utils import plot_train_test_split, plot_corr_matrix
import pandas as pd
import numpy as np
import os

class LTOPlotter:
    def __init__(self, prediction_task, output_dir, save=True, best_lambda=None):
        self.prediction_task = prediction_task
        if self.prediction_task == 'MethylationSubgroup':
            self.class_ids = ['Merlin Intact', 'Immune Enriched', 'Hypermetabolic']
        else:
            self.class_ids = ['Intact', 'Loss']
        self.n_classes = len(self.class_ids)

        if best_lambda is not None:
            self.best_lambda = best_lambda
            self.output_dir = f"{output_dir}/{prediction_task}/lambda_{best_lambda}"
        else:
            self.output_dir = f"{output_dir}/{prediction_task}"

        self.lto_evolution_dir = f"{self.output_dir}/evolution_curves"
        for d in [self.output_dir, self.lto_evolution_dir]:
            if not os.path.exists(d) and save: os.makedirs(d)
        
        self.save = save

    def set_best_lambda(self, best_lambda):
        self.best_lambda = best_lambda

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plots a confusion matrix given y_true labels and y_pred predictions and returns the matrix."""
        conf_matrix = confusion_matrix(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='viridis', cbar=False, xticklabels=self.class_ids, yticklabels=self.class_ids)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix (lambda = {round(self.best_lambda, 2)})\nBalanced Accuracy = {balanced_accuracy*100:.2f}%')
        
        if self.save:
            plt.savefig(f'{self.output_dir}/final_test_confusion_matrix.png')
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
            plt.savefig(f'{self.output_dir}/final_test_perf_metrics.png')
            plt.close()
            
            # Save the metrics to their own csv file
            metrics_df.to_csv(f'{self.output_dir}/final_test_perf_metrics.csv', index=False)
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
        plt.title(f'ROC Curve (lambda = {round(self.best_lambda, 2)})')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{self.output_dir}/final_test_roc_curve.png')
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
            title=f"{self.prediction_task}: One-vs-Rest ROC Curves (lambda = {round(self.best_lambda, 2)})",
        )

        if self.save: 
            plt.savefig(f'{self.output_dir}/final_test_roc_curve.png')
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

    def process_metrics(self, metrics_by_lambda, metric):
        # Processing Training Data
        means = {}
        stds = {}
        for key, df in metrics_by_lambda.items():
            if metric in df.columns:
                means[key] = df[metric].mean()
                stds[key] = df[metric].std()
            else:
                print(f"Metric {metric} not found in training data for lambda {key}")
        
        mean_df = pd.DataFrame(list(means.items()), columns=['Lambda', 'Mean'])
        std_df = pd.DataFrame(list(stds.items()), columns=['Lambda', 'Std Dev'])
        return mean_df, std_df

    def plot_metric_by_lambda(self, train_metrics_by_lambda, val_metrics_by_lambda, test_metrics_by_lambda, metric):
        # Processing Training & Validation Data
        train_mean_df, train_std_df = self.process_metrics(train_metrics_by_lambda, metric)
        val_mean_df, val_std_df = self.process_metrics(val_metrics_by_lambda, metric)

        # Processing Test Data
        test_metrics = {k: df[metric].iloc[0] for k, df in test_metrics_by_lambda.items() if metric in df.columns}
        test_df = pd.DataFrame(list(test_metrics.items()), columns=['Lambda', metric])

        # Pick best lambda and it's value to include in title
        max_perf_met_index = test_df[metric].idxmax()
        best_lambda = round(test_df.loc[max_perf_met_index, 'Lambda'], 2)
        best_value = round(test_df.loc[max_perf_met_index, metric], 3)

        plt.figure(figsize=(12, 8))

        # Plotting Training & Validation Data
        plt.errorbar(train_mean_df['Lambda'], train_mean_df['Mean'], yerr=train_std_df['Std Dev'], label=f'Train {metric}', marker='o')
        plt.errorbar(val_mean_df['Lambda'], val_mean_df['Mean'], yerr=val_std_df['Std Dev'], label=f'Validation {metric}', marker='s')

        # Plotting Test Data
        plt.plot(test_df['Lambda'], test_df[metric], label=f'Test {metric}', linestyle='--', marker='x')

        plt.xlabel('Lambda')
        plt.ylabel(metric)
        plt.title(f'{self.prediction_task}: LTO CV Evolution Curve\nBest Lambda = {best_lambda} by {metric} = {best_value}')
        plt.legend()
        plt.grid(True)
        if self.save:
            plt.savefig(f'{self.lto_evolution_dir}/{metric}.png')
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
        plt.title(f'{name} Heatmap of L1-regularized coefs across folds\n(lambda = {round(self.best_lambda, 2)})', fontsize=20)
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{self.output_dir}/{plot_name}_coefs_heatmap.png')
        else:
            plt.show()
        plt.close()

    def plot_coef_bargraph(self, data, name, plot_name=None):
        plt.figure(figsize=(12, 12))
        data = data.abs()
        data = data.iloc[::-1]
        data.plot(kind='barh', stacked=True, colormap='viridis', ax=plt.gca(), legend=False)
        plt.title(f'{name} Sum of Coef Absolute Values \n(lambda = {round(self.best_lambda, 2)})', fontsize=20)
        plt.ylabel('Feature', fontsize=16)
        plt.xlabel('Sum of Coef Absolute Values (coloured by fold #)', fontsize=16)
        plt.yticks(rotation=45, fontsize=12)
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{self.output_dir}/{plot_name}_coefs_bargraph.png')
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
            plt.savefig(f'{self.output_dir}/{plot_name}.png')
        else:
            plt.show()
        plt.close()

    def plot_coefs(self, coef_arr, intercept, feat_names):
        if self.n_classes > 2:
            for k, biomarker in enumerate(self.class_ids):
                coefs = pd.DataFrame(coef_arr[:, k, :], columns=feat_names)
                coefs['intercept'] = intercept[:, k]
                zero_feats = [col for col in coefs.columns if (coefs[col] == 0.).all()]
                nonzero_coefs = coefs.drop(columns=zero_feats)
                nonzero_coefs = nonzero_coefs.T

                if nonzero_coefs.empty:
                    print(f"No non-zero coefficients for {biomarker}")
                    continue
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

                if self.save: nonzero_coefs.to_csv(f'{self.output_dir}/{biomarker}_coefs.csv')

        else:
            coefs = pd.DataFrame(coef_arr.squeeze(), columns=feat_names)
            coefs['intercept'] = intercept
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

            if self.save: nonzero_coefs.to_csv(f'{self.output_dir}/{self.prediction_task}_coefs.csv')
        
        return nonzero_coefs
