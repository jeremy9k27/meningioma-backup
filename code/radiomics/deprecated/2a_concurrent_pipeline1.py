# Package imports
from Experiment import Experiment
import time
from datetime import datetime
import sys
import os
from joblib import Parallel, delayed

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup

setup()

# Define experiment parameters
USE_SMOTE = True
SCALER = 'Standard'
EVEN_TEST_SPLIT = True
OUTPUT_DIR = 'data/radiomics/debugging/logging'
N_JOBS = 2

# Define feature selection and final classification models to use
rfe_models = ['LDA', 'LinearSVM'] # ['LDA', 'LinearSVM', 'LogisticRegression', 'RandomForest', 'GradientBoosting', 'XGBoost']
final_models = ['LDA', 'GaussianProcess'] # ['LDA', 'GaussianProcess', 'SVM', 'LogisticRegression', 'RandomForest', 'GradientBoosting', 'XGBoost']

# Cross product
feature_selectors = rfe_models * len(final_models)
classifiers = [x for x in final_models for _ in range(len(rfe_models))]

N = len(feature_selectors)

# Define tasks, test set sizes, and seeds to loop thru
tasks = ['Chr22q', 'MethylationSubgroup'] # ['Chr22q', 'MethylationSubgroup', 'Chr1p'] 
test_sizes = [16, 18] # [16, 18, 16]
seeds = [22] # [22, 23, 24, 25, 26]

overall_begin_time = time.time()
overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def par_rfe(task, test_size, seed, feat_selector, model_index, total_models):
    try:
        # print(f"Feature selection model {model_index}/{total_models}: {feat_selector}")
        exp = Experiment(
            prediction_task=task, 
            test_size=test_size, 
            seed=seed, 
            feature_selection_model=feat_selector, 
            final_classifier_model='LDA', # dummy model, just needs to be valid 
            use_smote=USE_SMOTE,
            scaler=SCALER,
            even_test_split=EVEN_TEST_SPLIT,
            output_dir=OUTPUT_DIR,
            parallel_process_id=model_index,
            n_jobs=N_JOBS
        )
        exp.run_rfe()
    except Exception as e:
        pass
        # print(f"\n\nError during RFE using model {model_index}/{total_models}: {feat_selector};\n\n{e}")

def par_final(task, test_size, seed, feat_selector, classifier, model_index, total_models):
    try:
        # print(f"Model combo {model_index}/{total_models}: {feat_selector} + {classifier}")
        exp = Experiment(
            prediction_task=task, 
            test_size=test_size, 
            seed=seed, 
            feature_selection_model=feat_selector, 
            final_classifier_model=classifier, 
            use_smote=USE_SMOTE,
            scaler=SCALER,
            even_test_split=EVEN_TEST_SPLIT,
            output_dir=OUTPUT_DIR,
            parallel_process_id=model_index,
            n_jobs=N_JOBS
        )
        exp.run_all()
    except Exception as e:
        pass
        # print(f"\n\nError during Final model using model combo {model_index}/{total_models}: {feat_selector} + {classifier};\n\n{e}")

# Start loop
for task, test_size in zip(tasks, test_sizes):
    for seed in seeds:
        print(f"Starting task {task}, seed {seed}")
        # First run univariate feature selection (does not depend on any models)
        exp = Experiment(
            prediction_task=task, 
            test_size=test_size, 
            seed=seed, 
            feature_selection_model='LDA', # dummy model, just needs to be valid 
            final_classifier_model='LDA', # dummy model, just needs to be valid 
            use_smote=USE_SMOTE,
            scaler=SCALER,
            even_test_split=EVEN_TEST_SPLIT,
            output_dir=OUTPUT_DIR
        )
        exp.run_univariate()

        # Next, run RFE feature selection using each possible rfe model
        Parallel(n_jobs=N_JOBS)(delayed(par_rfe)(task, test_size, seed, feat_selector, i + 1, len(rfe_models)) for i, feat_selector in enumerate(rfe_models))
        
        # Lastly, run final classification using each model
        Parallel(n_jobs=N_JOBS)(delayed(par_final)(task, test_size, seed, feat_selector, classifier, j + 1, N) for j, (feat_selector, classifier) in enumerate(zip(feature_selectors, classifiers)))

overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
overall_time_elapsed = time.time() - overall_begin_time
hours, rem = divmod(overall_time_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
print("\n")
print(f'Completed at {overall_end_time}')
print(f'Total elapsed time: {overall_time_elapsed}\n')