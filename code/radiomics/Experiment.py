import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.feature_selection import RFE, RFECV, mutual_info_classif, chi2, f_classif, SelectKBest, SelectFpr, SelectFdr, SelectFwe
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import get_data, plot_corr_matrix
from preprocessing.utils import setup
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, jaccard_score
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import joblib
import os
from Models import TextureAnalysisModel

class Experiment:
    def __init__(self, prediction_task, test_size, seed, feature_selection_model='RandomForest', final_classifier_model='RandomForest', use_smote=False, scaler=None, even_test_split=False, rfe_step_size=64, output_dir='data/radiomics/evaluations/debugging', save=True):
        """
        Initialize the experiment with the provided settings. 
        
        Notes
        -----
        * scaler: The scaler to use for the experiment. If None, no scaling will be applied. Can be None, 'Standard', or 'MinMax'.
        * use_smote: Whether to use SMOTE for oversampling the minority class. If True, the scaler must be provided. Only makes sense for imbalanced datasets (i.e., NOT MethylationSubgroup).
        """
        # User specified settings
        self.prediction_task = prediction_task
        self.test_size = test_size
        self.seed = seed
        self.feat_select_model = TextureAnalysisModel(name=feature_selection_model)
        self.final_clf_model = TextureAnalysisModel(name=final_classifier_model)
        self.use_smote = use_smote
        self.scaler = scaler
        self.even_test_split = even_test_split
        self.rfe_step_size = rfe_step_size
        self.save = save

        # Settings we don't typically need to change
        self.exp_name = f"Scaler-{scaler}_SMOTE-{use_smote}_EvenTestSplit-{even_test_split}"
        self.feat_file = f"data/radiomics/features6/features_wide.csv" # File location with the radiomics features in wide format
        self.gs_params_size = 'small' # 'big' or 'small' depending on the size of the gridsearch
        self.num_MI_runs = 5 # 20 # Number of runs to perform for mutual information feature selection
        self.final_feat_set_size = [32, 64] # [2**x for x in range(5, 11)] # Number of features to use in the final classification model. One of: 32, 64, 128, 256, 512, 1024
        self.n_jobs = 4 # 4 # Number of jobs to run in parallel for gridsearches

        # Setting up the output directories
        self.output_dir = f"{output_dir}/{prediction_task}_TestSize-{test_size}/Seed-{seed}/{self.exp_name}"
        if not os.path.exists(self.output_dir) and save: os.makedirs(self.output_dir)

        self.output_univariate_fs = f"{self.output_dir}/univariate_fs"
        if not os.path.exists(self.output_univariate_fs) and save: os.makedirs(self.output_univariate_fs)

        self.output_rfe_fs = f"{self.output_dir}/rfe_fs/{feature_selection_model}"
        if not os.path.exists(self.output_rfe_fs) and save: os.makedirs(self.output_rfe_fs)

        self.output_final_model = f"{self.output_dir}/final_model/featselect-{feature_selection_model}_finalmodel-{final_classifier_model}"
        if not os.path.exists(self.output_final_model) and save: os.makedirs(self.output_final_model)

        # Automatically generated settings
        self.gs_summary_file = f"{self.output_dir}/gridsearch_summary.csv"
        self.results_summary_file = f"{self.output_dir}/results_summary.csv"
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

        # Objects to be set later
        self.train_subjects_df = None
        self.test_subjects_df = None
        self.uni_overall_df, self.uni_all_feats, self.uni_robust_feats = None, None, None
        self.rfe_feat_ranking = None
        self.current_k = None
        self.mode = None # to say whether we are evaluating on the training or test set

    def _uni_fs(self, X, y, method=mutual_info_classif, method_name='MI'):
        """
        Performs univariate feature selection using the specified method and return a dataframe with the feature scores and ranks.
        """
        selector = SelectKBest(method, k='all')
        selector = selector.fit(X, y)
        df = pd.DataFrame({'feat': X.columns, f'{method_name}': selector.scores_}).sort_values(by=f'{method_name}', ascending=False)
        df[f'{method_name}_rank'] = df[f'{method_name}'].rank(ascending=False)
        return df
    
    def _plot_feat_ranks_scatter(self, df, filename, sort_val_key='mean_rank', top_k=128, categorical=False):
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
        if self.save: 
            plt.savefig(f'{filename}')
        else:
            plt.show()
        plt.close()

    def _plot_feat_ranks_line(self, df, filename, sort_val_key = 'MImean_rank', std_key='MIstd', top_k=128):
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
        if self.save:
            plt.savefig(f'{filename}')
        else:
            plt.show()
        plt.close()

    def _get_uni_sets(self, df, top_k=1024):
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

    def _univariate_step(self, X, y):
        """
        Performs univariate feature selection using mutual information, Chi2, and ANOVA, and saves the results.
        """
        if os.path.exists(f"{self.output_univariate_fs}/overall_feats.csv"):
            print("\tUnivariate feature selection for this experiment already done and saved. Loading univariate feature selection results...")
            overall_uni_df = pd.read_csv(f"{self.output_univariate_fs}/overall_feats.csv")
        else:
            uni_df = None
            
            # Remove constant features
            constant_feats = [col for col in X.columns if X[col].nunique() == 1]
            X = X.drop(columns=constant_feats)

            # We will start with mutual information, which needs multiple runs to combat its sensitivity to random seed
            # We will run MI self.num_MI_runs-many times and take the mean of the scores.
            print(f"\tSubstep 1/3: Running MI {self.num_MI_runs} times to get a more robust estimate of feature importance...")
            for i in tqdm(range(self.num_MI_runs)):
                df = self._uni_fs(X, y, method=lambda X, y: mutual_info_classif(X=X, y=y, random_state=i), method_name=f'MI{i}')
                if uni_df is None:
                    uni_df = df
                else:
                    uni_df = uni_df.merge(df, on='feat', how='outer')
            # Now we have the MI score for each feature across all runs, we can take the mean & std dev and rank them
            # We will rank them based on the mean score alone. The std dev is for visualizing the spread of the scores.
            uni_df['MImean'] = uni_df[[f'MI{i}' for i in range(self.num_MI_runs)]].mean(axis=1)
            uni_df['MImean_rank'] = uni_df['MImean'].rank(ascending=False)
            uni_df['MIstd'] = uni_df[[f'MI{i}_rank' for i in range(self.num_MI_runs)]].std(axis=1)
            print("\tSubstep 2/3: MI done! Now plotting results and moving on to Chi2 and ANOVA...")
            # Let's plot the top 512 features by mean MI score
            self._plot_feat_ranks_scatter(uni_df, filename=f'{self.output_univariate_fs}/MI_scatter.png', top_k=512)
            self._plot_feat_ranks_line(uni_df, filename=f'{self.output_univariate_fs}/MI_line.png', top_k=512)

            # Now we will perform univariate feature selection using Chi2 and ANOVA
            methods = [chi2, f_classif]
            method_names = ['Chi2', 'ANOVA']
            # We need to scale the features for these methods - to preserve the original distribution of the data we will use MinMaxScaler
            X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
            for method, method_name in zip(methods, method_names):
                df = self._uni_fs(X_scaled, y, method=method, method_name=method_name)
                uni_df = uni_df.merge(df, on='feat', how='outer')
            overall_uni_df = uni_df[['feat', 'MImean_rank', 'Chi2_rank', 'ANOVA_rank']]
            # Chi2 and ANOVA may return NaNs for some features that don't have sufficient information, we will fill these with the middle rank (num features // 2)
            # before calculating the average rank across all univariate methods
            overall_uni_df.fillna(overall_uni_df.shape[0]//2, inplace=True)
            overall_uni_df['avg_rank'] = overall_uni_df[[col for col in overall_uni_df.columns if '_rank' in col]].mean(axis=1)
            overall_uni_df['std'] = overall_uni_df[[col for col in overall_uni_df.columns if '_rank' in col]].std(axis=1)
            overall_uni_df.sort_values(by='avg_rank', ascending=True, inplace=True)
            print("\tSubstep 3/3: Chi2 and ANOVA done! Now saving and plotting results...")

            # Save and plot the overall univariate feature selection results
            if self.save: overall_uni_df.to_csv(f'{self.output_univariate_fs}/overall_feats.csv', index=False)
            self._plot_feat_ranks_scatter(overall_uni_df, filename=f"{self.output_univariate_fs}/overall_scatter.png", sort_val_key='avg_rank')
            self._plot_feat_ranks_scatter(overall_uni_df, filename=f"{self.output_univariate_fs}/overall_categorical_scatter.png", sort_val_key='avg_rank', categorical=True)
            self._plot_feat_ranks_line(overall_uni_df, filename=f"{self.output_univariate_fs}/overall_line.png", sort_val_key='avg_rank', std_key='std')

        # Get sets of the top 1024 features from each univariate method
        _, all_uni_features, robust_uni_features = self._get_uni_sets(overall_uni_df, top_k=1024)
        print(f"\tNumber of robust features (common across all univariate selection methods): {len(robust_uni_features)}")
        print(f"\tNumber of unique features across all methods: {len(all_uni_features)}")

        return overall_uni_df, all_uni_features, robust_uni_features

    def _plot_rfecv_results(self, rfecv):
        """
        Plots the mean validation scores and standard deviations (error bars) for each number of features selected by RFECV.
        """
        feature_set_sizes = list(np.arange(rfecv.n_features_in_, 1, -(rfecv.n_features_in_ // self.rfe_step_size)))[::-1]
        n_scores = len(rfecv.cv_results_["mean_test_score"])
        if n_scores != len(feature_set_sizes):
            feature_set_sizes = [1] + feature_set_sizes
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean test accuracy")
        plt.errorbar(
            feature_set_sizes,
            rfecv.cv_results_["mean_test_score"],
            yerr=rfecv.cv_results_["std_test_score"],
        )
        plt.title(f"Recursive Feature Elimination \nOptimal={rfecv.n_features_}")
        if self.save:
            plt.savefig(f"{self.output_rfe_fs}/rfecv_results.png")
        else:
            plt.show()
        plt.close()

    def _rfe_step(self, X, y):
        """
        Performs recursive feature elimination using a classifier and saves the results.
        """
        # Gridsearch for a classifier to use during recursive feature elimination in next step
        print("\tSubstep 1/2: Performing gridsearch for classifier to use during RFE step...")
        if os.path.exists(f"{self.output_rfe_fs}/pre_rfe_clf_gs.joblib"):
            gs = joblib.load(f"{self.output_rfe_fs}/pre_rfe_clf_gs.joblib")
            print("\tGridsearch looks like it was already done, so results were loaded from the pre-existing file!")
        else:
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.seed)
            estimator = self.feat_select_model.model()
            if self.gs_params_size == 'small':
                params = self.feat_select_model.params_small
            else:
                params = self.feat_select_model.params_big

            # make sure if the feature selection model is SVM, we only use linear kernel, since that is the only one that provides feature importances
            if self.feat_select_model.name == 'SVM':
                params['kernel'] = ['linear']
            elif self.feat_select_model.name == 'XGBoost':
                params['scale_pos_weight'] = [sum(y == 0) / sum(y == 1), 1]
                if self.n_classes == 2:
                    params['objective'] = ['binary:logistic']
                else:
                    params['objective'] = ['multi:softmax']
            
            gs = GridSearchCV(
                estimator=estimator,
                param_grid=params,
                scoring='f1_macro',
                refit=False,
                n_jobs=self.n_jobs,
                cv=cv,
                return_train_score=True,
                verbose=1
            )

            gs.fit(X, y)

            if self.save: 
                print("\tGridsearch done! Saving results to file...")
                joblib.dump(gs, f"{self.output_rfe_fs}/pre_rfe_clf_gs.joblib")
                gs_summary_line = pd.DataFrame({'GS_name': ['pre_rfe_clf_gs'], 'Model_name': [f"FeatureSelection-{self.feat_select_model.name}"], 'Num_feats': [X.shape[1]], 'Top_f1_macro': [gs.best_score_], 'Best_params': [str(gs.best_params_)]})
                if os.path.exists(self.gs_summary_file): 
                    gs_summary_line.to_csv(self.gs_summary_file, mode='a', header=False, index=False)
                else:
                    gs_summary_line.to_csv(self.gs_summary_file, mode='w', index=False)

        print("\tBest parameters from pre RFE gridsearch: ", gs.best_params_)
        print("\tBest score from pre RFE gridsearch: ", gs.best_score_)

        # Cross-validation and recursive feature elimination
        print("\tSubstep 2/2: Performing cross-validated recursive feature elimination...")
        if os.path.exists(f"{self.output_rfe_fs}/rfecv.joblib"):
            rfecv = joblib.load(f"{self.output_rfe_fs}/rfecv.joblib")
            print("\tRFECV looks like it was already done, therefore results loaded from the pre-existing file!")
            print("\tOptimal number of features: ", rfecv.n_features_)
            print("\tOptimal score: ", max(rfecv.cv_results_['mean_test_score']))
            self._plot_rfecv_results(rfecv)
            sorted_feats = pd.read_csv(f"{self.output_rfe_fs}/rfe_feat_ranking.csv")['RFE_feat_ranking'].to_list()
        else:
            kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=self.seed)
            if self.feat_select_model.name == 'LDA':
                clf = self.feat_select_model.model(**gs.best_params_)
            else:
                clf = self.feat_select_model.model(**gs.best_params_, random_state=self.seed)
            rfecv = RFECV(estimator=clf, min_features_to_select=1, step=X.shape[1]//self.rfe_step_size, cv=kf, scoring='f1_macro', verbose=0, n_jobs=self.n_jobs)
            rfecv.fit(X, y)
            print("\tRFECV done!")
            print("\tOptimal number of features: ", rfecv.n_features_)
            print("\tOptimal score: ", max(rfecv.cv_results_['mean_test_score']))
            if self.save: 
                print("\tSaving results to file...")
                joblib.dump(rfecv, f"{self.output_rfe_fs}/rfecv.joblib")
                gs_summary_line = pd.DataFrame({'GS_name': ['rfe_cv'], 'Model_name': [f"FeatureSelection-{self.feat_select_model.name}"], 'Num_feats': [rfecv.n_features_], 'Top_f1_macro': [max(rfecv.cv_results_['mean_test_score'])], 'Best_params': [str(gs.best_params_)]})
                if os.path.exists(self.gs_summary_file): 
                    gs_summary_line.to_csv(self.gs_summary_file, mode='a', header=False, index=False)
                else:
                    gs_summary_line.to_csv(self.gs_summary_file, mode='w', index=False)
        
            print("\tNow plotting RFECV results...")
            self._plot_rfecv_results(rfecv)
            sorted_feat_indices = np.argsort(rfecv.ranking_)
            sorted_feats = X.columns[sorted_feat_indices]
            if self.save: 
                print("\tSaving sorted feature rankings to file...")
                pd.DataFrame({'RFE_feat_ranking': sorted_feats}).to_csv(f"{self.output_rfe_fs}/rfe_feat_ranking.csv", index=False)
        return sorted_feats

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plots a confusion matrix given y_true labels and y_pred predictions and returns the matrix."""
        conf_matrix = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='viridis', cbar=False, xticklabels=self.class_ids, yticklabels=self.class_ids)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix ({self.mode} set), Train/test split: {self.train_subjects_df.shape[0]}/{self.test_size}\nOverall Accuracy = {accuracy*100:.2f}%')
        
        if self.save:
            plt.savefig(f'{self.output_final_model}/{self.current_k}feats_{self.mode}_confusion_matrix.png')
        else:
            plt.show()
        plt.close()

        return conf_matrix, accuracy

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
            plt.savefig(f'{self.output_final_model}/{self.current_k}feats_{self.mode}_set_metrics_table.png')
            plt.close()

            # Save the metrics to the gs summary file
            metrics_line = pd.DataFrame({'GS_name': [f'final_clf_{self.mode}_set_performance_metrics'], 'Model_name': [f"FeatureSelection-{self.feat_select_model.name}_FinalModel-{self.final_clf_model.name}"], 'Num_feats': [self.current_k], 'Top_f1_macro': [None], 'Best_params': [str(metrics)]})
            if os.path.exists(self.gs_summary_file):
                metrics_line.to_csv(self.gs_summary_file, mode='a', header=False, index=False)
            else:
                metrics_line.to_csv(self.gs_summary_file, mode='w', index=False)
            
            # Save the metrics to their own csv file
            metrics_df.to_csv(f'{self.output_final_model}/{self.current_k}feats_{self.mode}_set_metrics.csv', index=False)
        else:
            plt.show()
            plt.close()
    
    def _plot_binary_results(self, probs, y_true):
        """Plot ROC curve, confusion matrix, and metrics table for binary classification tasks. Returns the ROC AUC score."""

        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        roc_auc = auc(fpr, tpr)

        if self.mode == 'test':
            # Plot 1/3: ROC curve:
            plt.plot(fpr, tpr, color='tab:orange', lw=2, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], color='tab:blue', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve on {self.mode} set (Train/test split: {self.train_subjects_df.shape[0]}/{self.test_size})')
            plt.legend(loc="lower right")
            
            plt.tight_layout()
            if self.save:
                plt.savefig(f'{self.output_final_model}/{self.current_k}feats_{self.mode}_roc_curve.png')
            else:
                plt.show()
            plt.close()

        # Plot 2/3: Confusion matrix
        y_pred = np.argmax(probs, axis=1)
        if self.mode == 'test':
            conf_matrix, accuracy = self._plot_confusion_matrix(y_true, y_pred)
        else:
            conf_matrix = confusion_matrix(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
        
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
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Binary Jaccard': jaccard_score(y_true, y_pred, average='binary'),
            'Weighted Jaccard': jaccard_score(y_true, y_pred, average='weighted')
        }

        self._plot_metrics_table(metrics)

        return metrics['Binary F1']

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

        if self.mode == 'test':
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
                title=f"One-vs-Rest ROC Curves: {self.prediction_task}, {self.mode} set (Train/test split: {self.train_subjects_df.shape[0]}/{self.test_size})",
            )

            if self.save: 
                plt.savefig(f'{self.output_final_model}/{self.current_k}feats_{self.mode}_roc_curve.png')
            else:
                plt.show()

            plt.close()

        # Plot 2/3: Confusion matrix
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(probs, axis=1)
        if self.mode == 'test':
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

    def _test_model(self, clf, X, y):
        """Tests a model (clf) on a dataset (X, y) and returns results."""
        probs = clf.predict_proba(X)
        if self.n_classes == 2:
            f1 = self._plot_binary_results(probs, y)
        else:
            f1 = self._plot_multiclass_results(probs, label_binarize(y, classes=np.arange(self.n_classes)))
        
        return probs, f1

    def _final_fit(self, X_train, y_train, X_test, y_test):
        """Fits the final model using the top k features from self.rfe_feat_ranking, varying k for k in e.g. [32, 64, 128, 256, 512, 1024]."""
        train_probs_dict = {}
        test_probs_dict = {}

        # Loop thru the different feature set sizes
        for i, k in enumerate(self.final_feat_set_size):
            # Select top k features
            self.current_k = k
            feat_set = self.rfe_feat_ranking[:self.current_k]
            X_train_k = X_train[feat_set]
            y_train_k = y_train
            X_test_k = X_test[feat_set]
            print(f"\tFitting final model number {i + 1}/{len(self.final_feat_set_size)} using {self.current_k} features...")

            # Smote
            if self.use_smote:
                print(f"\t\tUsing SMOTE! Before SMOTE, X_train_k: {X_train_k.shape}, y_train_k: {y_train_k.shape}")
                X_train_k, y_train_k = SMOTE(random_state=self.seed).fit_resample(X_train_k, y_train_k)
                print(f"\t\tAfter SMOTE, X_train_k: {X_train_k.shape}, y_train_k: {y_train_k.shape}")
            
            # Gridsearch thru classifier
            print("\t\tPerforming gridsearch for final classifier...")
            if os.path.exists(f"{self.output_final_model}/final_clf_top{self.current_k}_feats_gs.joblib"):
                gs = joblib.load(f"{self.output_final_model}/final_clf_top{self.current_k}_feats_gs.joblib")
                print("\t\tGridsearch looks like it was already done, so results loaded from pre-existing file!")
            else:
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.seed)
                estimator = self.final_clf_model.model()
                if self.gs_params_size == 'small':
                    params = self.final_clf_model.params_small
                else:
                    params = self.final_clf_model.params_big

                if self.feat_select_model.name == 'XGBoost':
                    params['scale_pos_weight'] = [sum(y_train_k == 0) / sum(y_train_k == 1), 1]
                    if self.n_classes == 2:
                        params['objective'] = ['binary:logistic']
                    else:
                        params['objective'] = ['multi:softmax']

                gs = GridSearchCV(
                    estimator=estimator,
                    param_grid=params,
                    scoring='f1_macro',
                    refit=True,
                    n_jobs=self.n_jobs,
                    cv=cv,
                    return_train_score=True,
                    verbose=1
                )

                gs.fit(X_train_k, y_train_k)

                if self.save: 
                    print("\t\tGridsearch done! Saving results to file...")
                    joblib.dump(gs, f"{self.output_final_model}/final_clf_top{self.current_k}_feats_gs.joblib")
                    gs_summary_line = pd.DataFrame({'GS_name': ['final_clf_gs'], 'Model_name': [f"FeatureSelection-{self.feat_select_model.name}_FinalModel-{self.final_clf_model.name}"], 'Num_feats': [self.current_k], 'Top_f1_macro': [gs.best_score_], 'Best_params': [str(gs.best_params_)]})
                    if os.path.exists(self.gs_summary_file): 
                        gs_summary_line.to_csv(self.gs_summary_file, mode='a', header=False, index=False)
                    else:
                        gs_summary_line.to_csv(self.gs_summary_file, mode='w', index=False)

            # Print best results from gridsearch
            print("\t\tBest validation score from gridsearch: ", gs.best_score_)
            print("\t\tBest parameters from gridsearch: ", gs.best_params_)

            # Get test set results
            self.mode = 'test'
            test_set_probs, test_set_f1_score = self._test_model(gs.best_estimator_, X_test_k, y_test)

            # Get train set results
            self.mode = 'train'
            train_set_probs, train_set_f1_score = self._test_model(gs.best_estimator_, X_train_k, y_train_k)

            # Save the results 
            train_probs_dict[f"{self.current_k}_feats"] = train_set_probs
            test_probs_dict[f"{self.current_k}_feats"] = test_set_probs
            overall_experiment_results_line = pd.DataFrame({'Final_model': [self.final_clf_model.name], 'FeatSelect_model': [self.feat_select_model.name], 'Num_feats': [self.current_k], 'Train_score': [train_set_f1_score], 'Val_score': [gs.best_score_], 'Test_score': [test_set_f1_score], 'Best_final_params': [str(gs.best_params_)]})
            if self.save:
                print("\t\tSaving further results to file...")
                if os.path.exists(self.results_summary_file):
                    overall_experiment_results_line.to_csv(self.results_summary_file, mode='a', header=False, index=False)
                else:
                    overall_experiment_results_line.to_csv(self.results_summary_file, mode='w', index=False)
                
                joblib.dump({'train': train_probs_dict, 'test': test_probs_dict}, f"{self.output_final_model}/train_test_probs_dict.joblib")
        
        overall_experiment_results_df = pd.read_csv(self.results_summary_file)

        return overall_experiment_results_df, {'train': train_probs_dict, 'test': test_probs_dict}

    def run(self):
        """Runs radiomics experiment with the provided settings in the constructor."""

        print("\nRunning radiomics experiment...\n")
        # print(f"Settings: {self.__dict__}")

        # Step 0/3: Preparing the data for the experiment
        print("Step 0/3: Loading in data for the experiment...")
        X_train_df, y_train, train_subject_nums, X_test_df, y_test, test_subject_nums = get_data(
            features_file=self.feat_file, 
            outcome=self.prediction_task, 
            test_size=self.test_size, 
            seed=self.seed, 
            even_test_split=self.even_test_split, 
            scaler_obj=self.scaler_obj, 
            output_dir=self.output_dir
        )

        self.train_subjects_df = pd.DataFrame({'subject_num': list(train_subject_nums), 'true_label': y_train})
        self.test_subjects_df = pd.DataFrame({'subject_num': list(test_subject_nums), 'true_label': y_test})
        self.train_subjects_df.to_csv(f"{self.output_dir}/train_subjects.csv", index=False)
        self.test_subjects_df.to_csv(f"{self.output_dir}/test_subjects.csv", index=False)
        print("Data loaded successfully!")

        # Step 1: Univariate Feature Selection
        print("Step 1/3: Univariate Feature Selection...")
        self.uni_overall_df, self.uni_all_feats, self.uni_robust_feats = self._univariate_step(X_train_df, y_train)
        print("Univariate Feature Selection done!")

        # Step 2: Recursive Feature Elimination
        print("Step 2/3: Recursive Feature Elimination...")
        X_reduced = X_train_df[list(self.uni_all_feats)]
        self.rfe_feat_ranking = self._rfe_step(X_reduced, y_train)
        print("Recursive Feature Elimination done!")

        # Step 3: Fit final model using top k features from self.rfe_feat_ranking, varying k
        print("Step 3/3: Fitting final model for varying numbers of features...")
        exp_results_df, probs_dict = self._final_fit(X_train=X_train_df[self.rfe_feat_ranking], y_train=y_train, X_test=X_test_df[self.rfe_feat_ranking], y_test=y_test)
        print("Final model fitting done!")
        print("Completed radiomics experiment! Exiting run method.")

        return exp_results_df, probs_dict
    
        # then it is simply about defining the model and param space, and running the experiment for each one
        # look thru results of each pairing for model with highest validation accuracy (pretend u dont have test set results)
        
        # then ensemble the different kinds of models together, examine train and test set score of the ensemble

        # still need to do nested approach instead of current approach..!

        # still need to incorporate collage features

        # also try active learning at this point
    
    def debug(self):
        X_train_df, y_train, train_subject_nums, X_test_df, y_test, test_subject_nums = get_data(
            features_file=self.feat_file, 
            outcome=self.prediction_task, 
            test_size=self.test_size, 
            seed=self.seed, 
            even_test_split=self.even_test_split, 
            scaler_obj=self.scaler_obj, 
            output_dir=self.output_dir
        )

        self.train_subjects_df = pd.DataFrame({'subject_num': list(train_subject_nums), 'true_label': y_train})
        self.test_subjects_df = pd.DataFrame({'subject_num': list(test_subject_nums), 'true_label': y_test})

        return train_subject_nums, test_subject_nums