from Experiment import Experiment
import time
from datetime import datetime
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.utils import setup

setup()

tasks = ['Chr22q']
test_sizes = [17]
# rfe_models = ['LinearSVM', 'LogisticRegression', 'LDA', 'GradientBoosting', 'RandomForest', 'XGBoost']
# final_models = ['SVM', 'LogisticRegression', 'LDA', 'GradientBoosting', 'GaussianProcess', 'XGBoost']
rfe_models = ['LinearSVM']
final_models = ['SVM']

overall_begin_time = time.time()
overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for task, test_size in zip(tasks, test_sizes):
    print("Task:", task)
    for fs_m, clf_m in zip(rfe_models, final_models):
        print(f"Model combo: {fs_m} + {clf_m}")
        try: 
            exp = Experiment(
                prediction_task=task, 
                test_size=test_size, 
                seed=2, 
                feature_selection_model=fs_m, 
                final_classifier_model=clf_m, 
                use_smote=True,
                scaler='Standard',
                even_test_split=True,
                output_dir=f'data/radiomics/evaluations/debug_one_by_one'
            )
            exp.run()
        except Exception as e:
            print("\n\nError:", e)

overall_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
overall_time_elapsed = time.time() - overall_begin_time
hours, rem = divmod(overall_time_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
overall_time_elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
print("\n")
print(f'Completed at {overall_end_time}')
print(f'Total elapsed time: {overall_time_elapsed}\n')