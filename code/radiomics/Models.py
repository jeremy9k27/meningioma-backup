from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct

class TextureAnalysisModel:
    def __init__(self, name='RandomForest'):
        self.name = name

        # Random Forest - for feature selection and for final classification
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
        
        # Support Vector Machine - for final classification only
        elif self.name == 'SVM':
            self.model = SVC
            self.params_big = {
                'C': [0.5, 1, 5],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced'],
                'probability': [True]
            }
            self.params_small = {
                'C': [1],
                'kernel': ['linear'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced'],
                'probability': [True]
            }
        
        # Linear SVM - for feature selection only
        elif self.name == 'LinearSVM':
            self.model = LinearSVC
            self.params_big = {
                'penalty': ['l1', 'l2'],
                'loss': ['squared_hinge'],
                'dual': ['auto'],
                'C': [0.1, 0.5, 1, 5, 10],
                'class_weight': ['balanced'],
                'max_iter': [1000, 5000]
            }
            self.params_small = {
                'penalty': ['l1', 'l2'],
                'loss': ['squared_hinge'],
                'dual': ['auto'],
                'C': [1],
                'class_weight': ['balanced'],
                'max_iter': [5000]
            }

        # XGBoost - for feature selection (gblinear) and for final classification
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
        # https://xgboost.readthedocs.io/en/stable/parameter.html
        # https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
        elif self.name == 'XGBoost':
            self.model = XGBClassifier
            self.params_big = {
                'n_estimators': [25, 50, 75, 100, 200], # Number of boosting rounds
                'max_depth': [0, 6, 10, 20], # Maximum tree depth for base learners
                'grow_policy': [0, 1], # Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow depth-wise. 1: favor splitting at nodes with highest loss change
                'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5, 0.8, 1], # boosting learning rate, eta
                'objective': ['binary:logistic'], # or 'multi:softmax'; Specify the learning task and the corresponding learning objective or a custom objective function to be used
                'booster': ['gbtree', 'gblinear', 'dart'], # Specify which booster to use: gbtree, gblinear or dart
                'tree_method': ['hist'], # The tree construction algorithm used in XGBoost
                # 'n_jobs': [1], # Number of parallel threads used to run XGBoost
                'gamma': [0, 1, 10], # Minimum loss reduction required to make a further partition on a leaf node of the tree
                'min_child_weight': [1, 3, 5], # worth searching thru ??
                'subsample': [0.8, 1.0], # worth searching thru?? Subsample ratio of the training instances
                'colsample_bytree': [0.8, 1.0], # worth searching thru?? Subsample ratio of columns when constructing each tree
                'scale_pos_weight': [1],
                'reg_alpha': [0, 0.5, 1, 2, 5], # L1 regularization term on weights
                'reg_lambda': [0, 0.5, 1, 2, 5], # L2 regularization term on weights
            }
            self.params_small = {
                'n_estimators': [75],
                'max_depth': [6],
                'learning_rate': [0.3],
                'booster': ['gbtree', 'dart'],
                'objective': ['binary:logistic'],
            }
        
        # Logistic Regression - for feature selection and for final classification
        elif self.name == 'LogisticRegression':
            self.model = LogisticRegression
            self.params_big = {
                'penalty': ['elasticnet', None],
                'C': [0.1, 0.5, 1, 5, 10],
                'class_weight': ['balanced'],
                'solver': ['saga'],
                'max_iter': [100, 200, 300, 400, 500],
                'multi_class': ['ovr', 'multinomial'],
                'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
            }
            self.params_small = {
                'penalty': ['elasticnet'],
                'C': [1],
                'class_weight': ['balanced'],
                'solver': ['saga'],
                'max_iter': [100],
                'multi_class': ['ovr'],
                'l1_ratio': [0, 1]
            }
        
        # Linear Discriminant Analysis - for feature selection and for final classification
        elif self.name == 'LDA':
            self.model = LDA
            self.params_big = {
                'solver': ['svd', 'lsqr'],
                'shrinkage': [None, 'auto'],
            }
            self.params_small = {
                'solver': ['svd', 'lsqr'],
                'shrinkage': [None],
            }
        
        # Gradient Boosting - for feature selection and for final classification
        elif self.name == 'GradientBoosting':
            self.model = GradientBoostingClassifier
            self.params_big = {
                'loss': ['log_loss', 'exponential'],
                'learning_rate': [0.01, 0.1, 0.3, 0.5],
                'n_estimators': [50, 100, 200, 400],
                'subsample': [0.8, 1.0],
                'criterion': ['friedman_mse'],
                'min_samples_split': [2, 4, 6],
                'max_depth': [None, 3, 6, 9],
                'max_features': ['sqrt', 'log2', None],                
            }
            self.params_small = {
                'loss': ['log_loss'],
                'learning_rate': [0.1],
                'n_estimators': [200],
                'subsample': [0.8, 1.0],
                'criterion': ['friedman_mse'],
                'min_samples_split': [4],
                'max_depth': [3],
                'max_features': ['sqrt'],                
            }
        
        # Gaussian Process - for final classification only
        elif self.name == 'GaussianProcess':
            self.model = GaussianProcessClassifier
            self.params_big = {
                'kernel': [1.0 * RBF(1.0), 1.0 * Matern(nu=0.5), 1.0 * Matern(nu=1.5), 1.0 * Matern(nu=2.5), 1.0 * RationalQuadratic(), 1.0 * DotProduct()],
                'n_restarts_optimizer': [5, 10],
                'max_iter_predict': [100, 300, 500]
            }
            self.params_small = {
                'kernel': [1.0 * RBF(1.0), 1.0 * Matern(nu=1.5)],
                'n_restarts_optimizer': [10],
                'max_iter_predict': [500]
            }
        else:
            raise ValueError(f'Invalid model name: {name}. Must be one of RandomForest, SVM, LinearSVM, XGBoost, LogisticRegression, LDA, GradientBoosting, or GaussianProcess')
    
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