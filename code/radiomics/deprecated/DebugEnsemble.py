import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import prep_data_for_loocv
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from plotting import LTOPlotter

def split_array(array, value):
    # Find the index of the specified value in the array
    index = np.where(array == value)[0][0]
    
    # Create one array with the specified value
    array_with_value = array[index:index+1]
    
    # Create another array with the remaining values
    remaining_array = np.concatenate([array[:index], array[index+1:]])
    
    return remaining_array, array_with_value

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
        
        self.lto_splits_dir = f"{self.output_dir}/splits"
        self.lto_train_splits_dir = f"{self.lto_splits_dir}/train"
        self.lto_val_splits_dir = f"{self.lto_splits_dir}/val"
        self.lto_test_splits_dir = f"{self.lto_splits_dir}/test"
        for d in [self.output_dir, self.lto_splits_dir, self.lto_train_splits_dir, self.lto_val_splits_dir, self.lto_test_splits_dir]:
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
            raw_scores = np.stack([self.sigmoid(X_query @ coef[:, c, :].T + intercept[:, c]) for c in range(coef.shape[1])]).T
            probs = raw_scores / raw_scores.sum(axis=1)[:, np.newaxis]

        prob_avg = np.mean(probs, axis=0)
        prob_std = np.std(probs, axis=0)
        final_pred = np.argmax(prob_avg)

        return prob_avg, prob_std, final_pred

    def model_fit_predict(self, split, lmda, outer_idx, inner_idx):
        # split the data according to given indices
        train_idx, test_idx = split
        X_train = self.X.iloc[train_idx]
        y_train = self.y[train_idx]
        X_test = self.X.iloc[test_idx]
        y_test = self.y[test_idx]

        X_train['subject_ID'].to_csv(f"{self.lto_train_splits_dir}/outeridx-{outer_idx}-inneridx-{inner_idx}.csv", header=False)
        X_test['subject_ID'].to_csv(f"{self.lto_val_splits_dir}/outeridx-{outer_idx}-inneridx-{inner_idx}.csv", header=False)
        X_train = X_train.drop(columns=['subject_ID'])
        X_test = X_test.drop(columns=['subject_ID'])

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

    def inner_loop(self, split, lmda, outer_idx):
        # split the data according to given indices
        train_val_idx, test_idx = split
        X_test = self.X.iloc[test_idx]
        y_test = self.y[test_idx]

        X_test['subject_ID'].to_csv(f"{self.lto_test_splits_dir}/outeridx-{outer_idx}.csv", header=False)

        inner_splits = [split_array(train_val_idx, num) for num in train_val_idx]

        # collect training and validation results
        results = []

        # Inner Loop defining validation set (sequential, ~1.5min in total?)
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self.model_fit_predict, i_split, lmda, outer_idx, inner_idx)
                       for inner_idx, i_split in enumerate(inner_splits)]
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
        X_test = X_test.drop(columns=['subject_ID'])
        test_probs, _, test_preds = self.ensemble(coefs, intercepts, X_test) # _ = test_stds, same shape as test_probs, namely (self.n_classes,)

        return (train_metrics, val_metrics, test_probs, test_preds, y_test)

    def process_lambda(self, lmda):
        """
        Given a fixed lambda (`lmda`), perform a parallelized LTO experiment and plot the test results.
        """
        results = []

        outer_splits = [split_array(np.arange(len(self.X)), i) for i in range(len(self.X))]

        # Outer loop defining test set
        with ProcessPoolExecutor(max_workers=32) as executor:
            # Create tasks for each train_val / test split (N many splits)
            futures = [executor.submit(self.inner_loop, o_split, lmda, outer_idx)
                       for outer_idx, o_split in enumerate(outer_splits)]

            # Use progress bar to monitor progress of each of the N many outer folds
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing lambda={lmda}"):
                results.append(future.result())

        # Unpack results from all N-many outer folds
        train_metrics, val_metrics, test_probs, test_preds, y_test = zip(*results)

        lambda_results = {
            'lambda': lmda,
            'train_metrics': pd.concat(train_metrics, ignore_index=True),
            'val_metrics': pd.DataFrame(list(val_metrics)),
            'test_probs': np.stack(test_probs).squeeze(),
            'test_preds': np.stack(test_preds).squeeze(),
            'y_test': np.stack(y_test).squeeze()
        }

        # plot final results
        if self.n_classes == 2:
            self.plotter._plot_binary_results(probs=lambda_results['test_probs'], y_true=lambda_results['y_test'])
        else:
            self.plotter._plot_multiclass_results(probs=lambda_results['test_probs'], y_true=lambda_results['y_test'])
        
        
