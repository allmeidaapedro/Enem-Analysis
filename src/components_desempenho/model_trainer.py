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
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
                              'artifacts' directory with the filename 'model_desempenho.pkl'.

    Example:
        config = ModelTrainerConfig()
        print(config.model_file_path)  # Output: 'artifacts/model_desempenho.pkl'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    model_file_path = os.path.join('artifacts', 'model_desempenho.pkl')


class ModelTrainer:
    '''
    This class is responsible for training and saving the best Lasso Regression model from modelling notebook.

    Attributes:
        model_trainer_config (ModelTrainerConfig): An instance of `ModelTrainerConfig` for configuration settings.

    Methods:
        apply_model_trainer(train_prepared, test_prepared):
            Trains the best Lasso Regression model using the provided prepared training, testing and validation data,
            and returns MAE, RMSE and R2 on the test set.

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
        Trains the best Lasso Regression model using the provided prepared training, testing and validation data, 
        the best hyperparameters found during the modelling notebook using validation set and bayesian optimization and returns the MAE, RMSE and R2 on the test set.

        Args:
            train_prepared (pd.DataFrame): The prepared training data.
            test_prepared (pd.DataFrame): The prepared testing data.
            val_prepared (pd.DataFrame): The prepared validation data.

        Returns:
            float: The MAE, RMSE and R2 of the best model on the test set.

        Raises:
            CustomException: If an error occurs during the training and evaluation process.
        '''

        try:
            logging.info('Split train, test and validation prepared sets.')
            
            X_train_prepared = train_prepared.drop(columns=['nota_geral'])
            X_test_prepared = test_prepared.drop(columns=['nota_geral'])
            X_val_prepared = val_prepared.drop(columns=['nota_geral']) 

            y_train = train_prepared['nota_geral'].copy()
            y_test = test_prepared['nota_geral'].copy()
            y_val = val_prepared['nota_geral'].copy()       
       
            X_train_prepared_full = pd.concat([X_train_prepared, X_val_prepared])
            y_train_full = pd.concat([y_train, y_val])

            logging.info('Started to train the best Lasso Regression model.')

            best_params = {'alpha': 0.5,
                            'precompute': False,
                            'max_iter': 1500,
                            'tol': 0.00010784803495028986,
                            'warm_start': False,
                            'selection': 'cyclic'}

            best_model = Lasso(**best_params)

            best_model.fit(X_train_prepared_full, y_train_full)

            logging.info('Saving the best model.')

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                object=best_model
            )

            logging.info('Best model MAE, RMSE and R2 on test set returned.')

            y_pred = best_model.predict(X_test_prepared)
            
            MAE = mean_absolute_error(y_test, y_pred)
            RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
            R2 = r2_score(y_test, y_pred)

            return MAE, RMSE, R2

        except Exception as e:
            raise CustomException(e, sys)