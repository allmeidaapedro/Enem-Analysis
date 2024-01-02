'''
This script aims to train and save the selected final model from the modelling notebook.
'''

'''
Importing the libraries
'''

# File handling.
import os
from dataclasses import dataclass

# Debugging and verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# Modelling.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Utils.
from src.artifacts_utils import save_object


@dataclass
class ModelTrainerConfig:
    '''
    Configuration class for model training.

    This data class holds configuration parameters related to model training. It includes attributes such as
    `model_file_path` that specifies the default path to save the trained model file.

    Attributes:
        model_file_path (str): The default file path for saving the trained model. By default, it is set to the
                              'artifacts' directory with the filename 'model_abstencao.pkl'.

    Example:
        config = ModelTrainerConfig()
        print(config.model_file_path)  # Output: 'artifacts/model_abstencao.pkl'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    model_file_path = os.path.join('artifacts', 'model_abstencao.pkl')


class ModelTrainer:
    '''
    This class is responsible for training and saving the best Logistic Regression model from modelling notebook.

    Attributes:
        model_trainer_config (ModelTrainerConfig): An instance of `ModelTrainerConfig` for configuration settings.

    Methods:
        apply_model_trainer(train_prepared, test_prepared):
            Trains the best Logistic Regression model using the provided prepared training, testing and validation data,
            and returns ROC AUC and classification report on the test set.

    '''

    def __init__(self) -> None:
        '''
        Initializes a new instance of the `ModelTrainer` class.

        Attributes:
            model_trainer_config (ModelTrainerConfig): An instance of `ModelTrainerConfig` for configuration settings.
        '''
        self.model_trainer_config = ModelTrainerConfig()
    
    
    def apply_model_trainer(self, train_prepared, test_prepared, val_prepared):
        '''
        Trains the best Logistic Regression model using the provided prepared training, testing and validation data, 
        the best hyperparameters found during the modelling notebook using validation set and bayesian optimization and returns the ROC AUC score, and classification report on the test set.

        Args:
            train_prepared (pd.DataFrame): The prepared training data.
            test_prepared (pd.DataFrame): The prepared testing data.
            val_prepared (pd.DataFrame): The prepared validation data.

        Returns:
            float: The ROC AUC score and classification report of the best model on the test set.

        Raises:
            CustomException: If an error occurs during the training and evaluation process.
        '''

        try:
            logging.info('Split train, test and validation prepared sets.')
            
            X_train_prepared = train_prepared.drop(columns=['abstencao'])
            X_test_prepared = test_prepared.drop(columns=['abstencao'])
            X_val_prepared = val_prepared.drop(columns=['abstencao']) 

            y_train = train_prepared['abstencao'].copy()
            y_test = test_prepared['abstencao'].copy()
            y_val = val_prepared['abstencao'].copy()       
       
            X_train_prepared_full = pd.concat([X_train_prepared, X_val_prepared])
            y_train_full = pd.concat([y_train, y_val])

            logging.info('Started to train the best Logistic Regression model.')

            best_params = {'penalty': 'l1', 
               'tol': 0.08132164711416587, 
               'C': 0.49107577496414995, 
               'max_iter': 2500, 
               'warm_start': False,
               'n_jobs': -1,
               'class_weight': 'balanced',
               'solver': 'saga'}
            
            best_model = LogisticRegression(**best_params)

            best_model.fit(X_train_prepared_full, y_train_full)

            logging.info('Saving the best model.')

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                object=best_model
            )

            logging.info('Best model ROC AUC, and classification report on test set returned.')

            y_pred = best_model.predict(X_test_prepared)
            probas = best_model.predict_proba(X_test_prepared)[:, 1]

            roc_auc = roc_auc_score(y_test, probas)
            class_report = classification_report(y_test, y_pred)

            return roc_auc, class_report

        except Exception as e:
            raise CustomException(e, sys)