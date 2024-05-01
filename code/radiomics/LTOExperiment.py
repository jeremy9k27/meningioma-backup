import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import plot_train_test_split, plot_corr_matrix, prep_data_for_loocv, split_array
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from plotting import LTOPlotter

class LTOExperiment:
    def __init__(self, prediction_task, lambdas=[0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15], use_smote=True, scaler='Standard', seed=0, output_dir='data/lto', save=True, debug=False, best_lambda=None):
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
        if best_lambda is not None:
            self.best_lambda = best_lambda
            self.output_dir = f"{output_dir}/{prediction_task}/lambda_{best_lambda}"
        else:
            self.output_dir = f"{output_dir}/{prediction_task}"
        
        self.lto_evolution_dir = f"{self.output_dir}/evolution_curves"
        for d in [self.output_dir, self.lto_evolution_dir]:
            if not os.path.exists(d) and save: os.makedirs(d)
        
        self.save = save
        
        # make plotter object to handle plotting, has all the custom plotting functions needed for experiment.
        self.plotter = LTOPlotter(prediction_task=prediction_task, output_dir=output_dir, save=save, best_lambda=best_lambda)

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

        if debug: self.X = self.X[:15]

    def sigmoid(self, x): return 1/(1 + np.exp(-x)) 

    def ensemble(self, coef, intercept, X_query):
        if self.n_classes == 2:
            raw_scores = self.sigmoid(X_query @ coef.T + intercept) # (N, 1)
            probs = np.vstack([1 - raw_scores, raw_scores]).T # (N, 2)
        else:
            raw_scores = np.stack([self.sigmoid(X_query @ coef[:, c, :].T + intercept[:, c]) for c in range(coef.shape[1])]).T.squeeze()
            probs = raw_scores / raw_scores.sum(axis=1)[:, np.newaxis]

        prob_avg = np.mean(probs, axis=0)
        prob_std = np.std(probs, axis=0)
        final_pred = np.argmax(prob_avg)

        return prob_avg, prob_std, final_pred

    def model_fit_predict(self, split, lmda):
        # split the data according to given indices
        train_idx, test_idx = split

        X_train = self.X.iloc[train_idx]
        y_train = self.y[train_idx]
        X_test = self.X.iloc[test_idx]
        y_test = self.y[test_idx]

        to_sub = len(self.X) - len(X_train)
        if self.use_smote:
            X_train, y_train = SMOTE(random_state=self.seed).fit_resample(X_train, y_train)

        model = LogisticRegression(C=lmda, **self.lr_params)
        model.fit(X_train, y_train)

        coefs = model.coef_
        intercepts = model.intercept_
        
        train_probs = model.predict_proba(X_train[:len(self.X)-to_sub])
        train_preds = model.predict(X_train[:len(self.X)-to_sub])

        test_probs = model.predict_proba(X_test)
        test_preds = model.predict(X_test)

        if self.n_classes == 2:
            train_metrics = self.plotter.get_binary_metrics(train_probs, train_preds, y_train[:len(self.X)-to_sub])
        else:
            train_metrics = self.plotter.get_multiclass_metrics(train_probs, train_preds, label_binarize(y_train[:len(self.X)-to_sub], classes=np.arange(self.n_classes)))

        return (coefs, intercepts, test_probs, test_preds, y_test, train_metrics)

    def inner_loop(self, split, lmda):
        # split the data according to given indices
        train_val_idx, test_idx = split
        X_train_val = self.X.iloc[train_val_idx]
        X_test = self.X.iloc[test_idx]
        y_test = self.y[test_idx]

        inner_splits = [split_array(train_val_idx, num) for num in train_val_idx]

        # collect training and validation results
        results = []

        # Inner Loop (sequential, ~1.5min in total?)
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self.model_fit_predict, i_split, lmda)
                       for i_split in inner_splits]
            for future in as_completed(futures):
                results.append(future.result())
        
        coefs, intercepts, val_probs, val_preds, y_val, train_metrics = zip(*results)
        
        # These are the training results, don't need to touch these
        train_metrics = pd.DataFrame(list(train_metrics))

        # Use these to get val_metrics
        val_probs = np.stack(val_probs).squeeze()
        val_preds = np.stack(val_preds).squeeze()
        y_val = np.stack(y_val).squeeze()
        # Actually get val_metrics
        if self.n_classes == 2:
            val_metrics = self.plotter.get_binary_metrics(val_probs, val_preds, y_val)
        else:
            val_metrics = self.plotter.get_multiclass_metrics(val_probs, val_preds, label_binarize(y_val, classes=np.arange(self.n_classes)))
        
        # Use these two to get test_probs, and test_preds
        coefs = np.stack(coefs).squeeze()
        intercepts = np.stack(intercepts).squeeze()
        # Actually get test_probs, and test_preds
        test_probs, _, test_preds = self.ensemble(coefs, intercepts, X_test) # _ = test_stds, same shape as test_probs, namely (self.n_classes,)

        return (train_metrics, val_metrics, test_probs, test_preds, y_test)

    def process_lambda(self, lmda):
        results = []

        outer_splits = [split_array(np.arange(len(self.X)), i) for i in range(len(self.X))]

        # Outer loop defining test set (Parallel, ~1.5min/fold, ~72folds, ~3.5min total?)
        with ProcessPoolExecutor(max_workers=32) as executor:
            # Create tasks for each train-test split
            futures = [executor.submit(self.inner_loop, o_split, lmda)
                       for o_split in outer_splits]

            # Use tqdm to monitor progress
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing lambda={lmda}"):
                results.append(future.result())

        # Unpack results
        train_metrics, val_metrics, test_probs, test_preds, y_test = zip(*results)

        return {
            'lambda': lmda,
            'train_metrics': pd.concat(train_metrics, ignore_index=True),
            'val_metrics': pd.DataFrame(list(val_metrics)),
            'test_probs': np.stack(test_probs).squeeze(),
            'test_preds': np.stack(test_preds).squeeze(),
            'y_test': np.stack(y_test).squeeze()
        }

    def final_model_loo(self, max_workers=1):
        results = []
        splits = [split_array(np.arange(len(self.X)), i) for i in range(len(self.X))]

        # Final Model (sequential, ~1.5min?)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.model_fit_predict, split, self.best_lambda)
                       for split in splits]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing final fit w/best lambda={self.best_lambda}"):
                results.append(future.result())

        # Unpack results
        coefs, intercepts, test_probs, test_preds, y_test, train_metrics = zip(*results)

        final_model_dict = {
            'coefs': np.stack(coefs).squeeze(),
            'intercepts': np.stack(intercepts).squeeze(),
            'test_probs': np.stack(test_probs).squeeze(),
            'test_preds': np.stack(test_preds).squeeze(),
            'y_test': np.stack(y_test).squeeze(),
            'train_metrics': pd.DataFrame(list(train_metrics))
        }
    
        self.coef = final_model_dict['coefs']
        self.intercept = final_model_dict['intercepts']
        self.test_probs = final_model_dict['test_probs']
        self.y_test = final_model_dict['y_test']

        self.nonzero_coefs = self.plotter.plot_coefs(coef_arr=self.coef, intercept=self.intercept, feat_names=self.feat_names)
        if self.n_classes > 2:
            self.plotter._plot_multiclass_results(self.test_probs, label_binarize(self.y_test, classes=np.arange(self.n_classes)))
        else:
            self.plotter._plot_binary_results(self.test_probs, self.y_test)

        self.final_model_dict = final_model_dict

    def par_lto_model(self, pmetric='AUC'):
        # Preparing storage dictionaries
        test_probs_by_lambda = {}
        y_tests_by_lambda = {}
        train_metrics_by_lambda = {}
        val_metrics_by_lambda = {}
        test_metrics_by_lambda = {}
        
        # Lambda Loop (sequential, ~3.5min/lambda?)
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(self.process_lambda, lmda): lmda for lmda in self.lambdas}
            for future in as_completed(futures):
                result = future.result()
                lmda = result['lambda']
                train_metrics_by_lambda[lmda] = result['train_metrics']
                val_metrics_by_lambda[lmda] = result['val_metrics']

                test_probs_by_lambda[lmda] = result['test_probs']
                test_preds = result['test_preds']
                y_tests_by_lambda[lmda] = result['y_test']
                if self.n_classes == 2:
                    test_metrics = pd.DataFrame(self.plotter.get_binary_metrics(result['test_probs'], test_preds, result['y_test']), index=[0])
                else:
                    test_metrics = pd.DataFrame(self.plotter.get_multiclass_metrics(result['test_probs'], test_preds, label_binarize(result['y_test'], classes=np.arange(self.n_classes))), index=[0])

                test_metrics_by_lambda[lmda] = test_metrics
        
        self.perf_metric = pmetric
        test_perf_met_by_lambda = self.plotter.plot_metric_by_lambda(train_metrics_by_lambda, val_metrics_by_lambda, test_metrics_by_lambda, pmetric)
        for metric in test_metrics_by_lambda[self.lambdas[0]].columns:
            self.plotter.plot_metric_by_lambda(train_metrics_by_lambda, val_metrics_by_lambda, test_metrics_by_lambda, metric)

        max_perf_met_index = test_perf_met_by_lambda[pmetric].idxmax()
        self.best_lambda = test_perf_met_by_lambda.loc[max_perf_met_index, 'Lambda']

        print(f"Best lambda: {round(self.best_lambda, 2)}")

        self.plotter.set_best_lambda(self.best_lambda)
        
        self.final_model_loo()
        self.train_metrics_by_lambda = train_metrics_by_lambda
        self.val_metrics_by_lambda = val_metrics_by_lambda
        self.test_metrics_by_lambda = test_metrics_by_lambda
