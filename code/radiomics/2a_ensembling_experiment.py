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

tasks = ['MethylationSubgroup', 'Chr22q', 'Chr1p']
test_sizes = [22, 17, 16]
seeds = [10, 11, 12, 13, 14]
models = ['RandomForest', 'SVM']
use_smote = [False, True]
even_test_splits = [False, True]

overall_begin_time = time.time()
overall_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for task, test_size in zip(tasks, test_sizes):
    print("Task:", task)
    for seed in seeds:
        for model in models:
            for smote in use_smote:
                for ets in even_test_splits:
                    print("\tSeed:", seed)
                    print("\t\tModel:", model)
                    print("\t\t\tSMOTE:", smote)
                    print("\t\t\tEven Test Splits:", ets)
                    try:
                        exp = Experiment(
                            prediction_task=task, 
                            test_size=test_size, 
                            seed=seed, 
                            feature_selection_model=model, 
                            final_classifier_model=model, 
                            use_smote=smote,
                            scaler='Standard',
                            even_test_split=ets
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