# Package imports
from code.radiomics.deprecated.Experiment import Experiment
import time
from datetime import datetime
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup

setup()

# Define experiment parameters
USE_SMOTE = True
SCALER = 'Standard'
EVEN_TEST_SPLIT = True
OUTPUT_DIR = 'data/radiomics_results_big_binary'
N_JOBS = 16

# Define feature selection and final classification models to use
rfe_models = ['LDA', 'LinearSVM', 'LogisticRegression', 'RandomForest', 'GradientBoosting', 'XGBoost'] # ['LDA', 'LinearSVM']
final_models = ['LDA', 'GaussianProcess', 'SVM', 'LogisticRegression', 'RandomForest', 'GradientBoosting', 'XGBoost'] # ['LDA', 'GaussianProcess']

# Cross product
feature_selectors = rfe_models * len(final_models)
classifiers = [x for x in final_models for _ in range(len(rfe_models))]

N = len(feature_selectors)

# Define tasks, test set sizes, and seeds to loop thru
tasks = ['Chr22q', 'Chr1p'] # ['Chr22q', 'MethylationSubgroup', 'Chr1p'] 
test_sizes = [16, 16] # [16, 18, 16]
seeds = [22, 23, 24, 25, 26]

overall_begin_time = time.time()
overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Start loop
for seed in seeds:
    for task, test_size in zip(tasks, test_sizes):
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
        for i, feat_selector in enumerate(rfe_models):
            try:
                print(f"Feature selection model {i + 1}/{len(rfe_models)}: {feat_selector}")
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
                    parallel_process_id=i + 1,
                    n_jobs=N_JOBS
                )
                exp.run_rfe()
            except Exception as e:
                print(f"\n\nError during RFE using model {i + 1}/{len(rfe_models)}: {feat_selector};\n\n{e}")
        
        # Lastly, run final classification using each model
        for j, (feat_selector, classifier) in enumerate(zip(feature_selectors, classifiers)):
            print(f"Model combo {j + 1}/{N}: {feat_selector} + {classifier}")
            try: 
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
                    parallel_process_id=j + 101,
                    n_jobs=N_JOBS
                )
                exp.run_all()
            except Exception as e:
                print(f"\n\nError during Final model using model combo {j + 1}/{N}: {feat_selector} + {classifier};\n\n{e}")

overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
overall_time_elapsed = time.time() - overall_begin_time
hours, rem = divmod(overall_time_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
print("\n")
print(f'Completed at {overall_end_time}')
print(f'Total elapsed time: {overall_time_elapsed}\n')