from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

class TextureAnalysisModel:
    def __init__(self, name='RandomForest'):
        self.name = name
        if self.name == 'RandomForest':
            self.model = RandomForestClassifier
            self.params_big = {
                'n_estimators': [25, 50, 75, 100, 200],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 4, 6],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            self.params_small = {
                'n_estimators': [100],
                'criterion': ['entropy'],
                'max_depth': [None, 10],
                'min_samples_split': [6],
                'max_features': ['sqrt'],
                'bootstrap': [True],
                'class_weight': ['balanced']
            }
        elif self.name == 'SVM':
            self.model = SVC
            self.params_big = {
                'C': [0.1, 1, 10, 100, 1000],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced']
            }
            self.params_small = {
                'C': [1],
                'kernel': ['linear'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced']
            }
        elif self.name == 'XGBoost':
            self.model = XGBClassifier
            self.params_big = {
                'n_estimators': [25, 50, 75, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3, 0.5],
                'max_depth': [3, 6, 9],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'objective': ['binary:logistic'],
                'nthread': [4],
                'scale_pos_weight': [1],
                'seed': [27]
            }
            self.params_small = {
                'n_estimators': [100],
                'learning_rate': [0.1],
                'max_depth': [3],
                'min_child_weight': [1],
                'gamma': [0],
                'subsample': [1.0],
                'colsample_bytree': [1.0],
                'objective': ['binary:logistic'],
                'nthread': [4],
                'scale_pos_weight': [1],
                'seed': [27]
            }
        elif self.name == 'LogisticRegression':
            self.model = LogisticRegression
            self.params_big = {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.1, 1, 10, 100, 1000],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 200, 300, 400, 500],
                'class_weight': ['balanced']
            }
            self.params_small = {
                'penalty': ['l1', 'elasticnet'],
                'C': [1],
                'solver': ['saga'],
                'max_iter': [100],
                'class_weight': ['balanced']
            }
        elif self.name == 'LDA':
            self.model = LDA
            self.params_big = {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto'],
                'n_components': [None, 1, 2, 3]
            }
            self.params_small = {
                'solver': ['svd', 'eigen'],
                'shrinkage': [None],
                'n_components': [None]
            }
        elif self.name == 'GradientBoosting':
            self.model = GradientBoostingClassifier
            self.params_big = {
                'n_estimators': [25, 50, 75, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3, 0.5],
                'max_depth': [3, 6, 9],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 3, 5],
                'max_features': ['sqrt', 'log2', None],
                'subsample': [0.8, 1.0],
                'criterion': ['friedman_mse', 'mse', 'mae'],
                'warm_start': [True, False]
            }
            self.params_small = {
                'n_estimators': [100],
                'learning_rate': [0.1, 0.3],
                'max_depth': [3],
                'min_samples_split': [6],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'subsample': [1.0],
                'criterion': ['friedman_mse'],
                'warm_start': [False]
            }
        elif self.name == 'GaussianProcess':
            self.model = GaussianProcessClassifier
            self.params_big = {
                'n_restarts_optimizer': [0, 1, 2, 3, 4, 5],
                'max_iter_predict': [100, 200, 300, 400, 500],
                'warm_start': [True, False]
            }
            self.params_small = {
                'n_restarts_optimizer': [0],
                'max_iter_predict': [100],
                'warm_start': [False]
            }
        else:
            raise ValueError(f'Invalid model name: {name}. Must be one of RandomForest, SVM, XGBoost, LogisticRegression, LDA, GradientBoosting, GaussianProcess.')
    
    def get_model(self):
        return self.model

    def get_params(self, big=False):
        if big:
            return self.params_big
        else:
            return self.params_small
    
    def set_model(self, model):
        self.model = model
    
    def set_params(self, params, big=False):
        if big:
            self.params_big = params
        else:
            self.params_small = params