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
import os

class Experiment:
    def __init__(self, prediction_task, test_size, seed, use_smote=False, scaler=None, even_test_split=False, output_dir='evaluations6', save=True):
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
        self.use_smote = use_smote
        self.scaler = scaler
        self.even_test_split = even_test_split
        self.save = save

        # Settings we don't typically need to change
        self.exp_name = f"{prediction_task}_TestSize-{test_size}_Seed-{seed}_SMOTE-{use_smote}_Scaler-{scaler}_EvenTestSplit-{even_test_split}"
        self.feat_file = f"data/radiomics/features5/features_wide.csv"
        
        # Setting up the output directories
        self.output_dir = f"data/radiomics/evaluations/{output_dir}/{self.exp_name}"
        if not os.path.exists(self.output_dir) and save: os.makedirs(self.output_dir)
        self.output_univariate_fs = f"{self.output_dir}/univariate_fs"
        if not os.path.exists(self.output_univariate_fs) and save: os.makedirs(self.output_univariate_fs)
        self.output_rfe_fs = f"{self.output_dir}/rfe_fs"
        if not os.path.exists(self.output_rfe_fs) and save: os.makedirs(self.output_rfe_fs)
        self.output_final_model = f"{self.output_dir}/final_model"
        if not os.path.exists(self.output_final_model) and save: os.makedirs(self.output_final_model)

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
        
        # Objects to be set later
        self.train_subjects_df = None
        self.test_subjects_df = None
        self.uni_overall_df, self.uni_all_feats, self.uni_robust_feats = None, None, None
        self.rfe_feat_ranking = None

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

            # We will start with mutual information, which needs multiple runs to combat its sensitivity to random seed
            # We will run MI 20 times and take the mean of the scores.
            num_MI_runs = 20
            print(f"\tRunning MI {num_MI_runs} times to get a more robust estimate of feature importance...")
            for i in tqdm(range(num_MI_runs)):
                df = self._uni_fs(X, y, method=lambda X, y: mutual_info_classif(X=X, y=y, random_state=i), method_name=f'MI{i}')
                if uni_df is None:
                    uni_df = df
                else:
                    uni_df = uni_df.merge(df, on='feat', how='outer')
            # Now we have the MI score for each feature across all runs, we can take the mean & std dev and rank them
            # We will rank them based on the mean score alone. The std dev is for visualizing the spread of the scores.
            uni_df['MImean'] = uni_df[[f'MI{i}' for i in range(num_MI_runs)]].mean(axis=1)
            uni_df['MImean_rank'] = uni_df['MImean'].rank(ascending=False)
            uni_df['MIstd'] = uni_df[[f'MI{i}_rank' for i in range(num_MI_runs)]].std(axis=1)
            print("\tMI done! Now plotting results and moving on to Chi2 and ANOVA...")
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
            print("\tChi2 and ANOVA done! Now saving and plotting results...")

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
        if self.save:
            plt.savefig(f"{self.output_rfe_fs}/rfecv_results.png")
        else:
            plt.show()
        plt.close()

    def _rfe_step(self, X, y):
        """
        Performs recursive feature elimination using a RF classifier and saves the results.
        """
        # Gridsearch for a RF classifier to use during recursive feature elimination in next step
        if os.path.exists(f"{self.output_rfe_fs}/classifier_gs_pre_rfe.joblib"):
            gs = joblib.load(f"{self.output_rfe_fs}/classifier_gs_pre_rfe.joblib")
            print("\tGridsearch results loaded from file.")
        else:
            print("\tPerforming gridsearch for RF classifier to use during RFE step...")
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.seed)
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
                refit=False,
                n_jobs=4,
                cv=cv,
                return_train_score=True,
                verbose=10
            )

            gs.fit(X, y)

            if self.save: joblib.dump(gs, f"{self.output_rfe_fs}/classifier_gs_pre_rfe.joblib")

        print("\tBest parameters from pre RFE gridsearch: ", gs.best_params_)
        print("\tBest score from pre RFE gridsearch: ", gs.best_score_)

        # Cross-validation and recursive feature elimination
        if os.path.exists(f"{self.output_rfe_fs}/rfecv.joblib"):
            rfecv = joblib.load(f"{self.output_rfe_fs}/rfecv.joblib")
            print("\tRFECV results loaded from file.")
        else:
            print("\tPerforming cross-validated recursive feature elimination...")
            kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=self.seed)
            clf = RandomForestClassifier(**gs.best_params_, random_state=self.seed)
            rfecv = RFECV(estimator=clf, min_features_to_select=1, step=len(X.shape[1])//32, cv=kf, scoring='f1_macro', verbose=3, n_jobs=4)
            rfecv.fit(X, y)
            if self.save: joblib.dump(rfecv, f"{self.output_rfe_fs}/rfecv.joblib")
        
        print("\tOptimal number of features: ", rfecv.n_features_)
        self._plot_rfecv_results(rfecv)
        sorted_feat_indices = np.argsort(rfecv.ranking_)
        sorted_feats = X.columns[sorted_feat_indices]
        return sorted_feats

    def run(self):
        """Runs radiomics experiment with the provided settings in the constructor."""

        # Step 0: Preparing the data for the experiment
        print("Step 0: Loading in data for the experiment...")
        X_train_df, y_train, X_test_df, y_test = get_data(
            features_file=self.feat_file, 
            outcome=self.prediction_task, 
            test_size=self.test_size, 
            seed=self.seed, 
            even_test_split=self.even_test_split, 
            scaler_obj=self.scaler_obj, 
            output_dir=self.output_dir
        )
        self.train_subjects_df = pd.DataFrame({'subject_num': list(X_train_df.index), 'true_label': y_train})
        self.test_subjects_df = pd.DataFrame({'subject_num': list(X_test_df.index), 'true_label': y_test})
        print("\tData loaded successfully!")

        # Step 1: Univariate Feature Selection
        print("Step 1: Univariate Feature Selection...")
        self.uni_overall_df, self.uni_all_feats, self.uni_robust_feats = self._univariate_step(X_train_df, y_train)
        print("\tUnivariate Feature Selection done!")

        # Step 2: Recursive Feature Elimination
        print("Step 2: Recursive Feature Elimination...")
        X_reduced = X_train_df[self.uni_all_feats]
        self.rfe_feat_ranking = self._rfe_step(X_reduced, y_train)
        print("\tRecursive Feature Elimination done!")
