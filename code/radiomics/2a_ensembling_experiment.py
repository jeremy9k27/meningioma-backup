from Experiment import Experiment

tasks = ['MethylationSubgroup', 'Chr22q']
test_sizes = [22, 17]
seeds = [0, 1]
models = ['RandomForest', 'SVM']

for task, test_size in zip(tasks, test_sizes):
    print("Task:", task)
    for seed in seeds:
        for model in models:
            print("\tSeed:", seed)
            print("\t\tModel:", model)
            try:
                exp = Experiment(
                    prediction_task=task, 
                    test_size=test_size, 
                    seed=seed, 
                    feature_selection_model=model, 
                    final_classifier_model=model, 
                    scaler='Standard'
                )
                exp.run()
            except Exception as e:
                print("\t\t\tError:", e)
            