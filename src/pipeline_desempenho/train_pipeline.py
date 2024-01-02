'''
This script aims to execute the entire training pipeline using the components. Specifically, data ingestion, transformation and model training.
'''

# Debugging and verbose.
import sys
from src.logger import logging
from src.exception import CustomException

# Components for data ingestion, transformation and model training.

# Data ingestion.
from src.components_desempenho.data_ingestion import DataIngestionConfig
from src.components_desempenho.data_ingestion import DataIngestion

# Data transformation.
from src.components_desempenho.data_transformation import DataTransformationConfig
from src.components_desempenho.data_transformation import DataTransformation

# Model trainer.
from src.components_desempenho.model_trainer import ModelTrainerConfig
from src.components_desempenho.model_trainer import ModelTrainer


class TrainPipeline:
    '''TrainPipeline class for training a machine learning pipeline.

    This class handles the training pipeline, including data ingestion,
    data transformation, and model training.

    Attributes:
        None

    Methods:
        __init__: Constructor method to initialize the class.
        train: Method to execute the training pipeline.

    '''

    def __init__(self) -> None:
        pass

    def train(self):
        '''Execute the training pipeline.

        This method performs the following steps:
        1. Data ingestion to obtain training, testing and validation datasets.
        2. Data transformation for preprocessing.
        3. Model training using the best hyperparameters.

        It also prints the MAE, RMSE and R2 scores.

        Raises:
            CustomException: If an error occurs during the training pipeline.

        '''

        try:
            logging.info('Train full pipeline started - Performance Model.')

            logging.info('Data Ingestion component started.')

            data_ingestion = DataIngestion()
            train_path, test_path, val_path = data_ingestion.apply_data_ingestion()

            logging.info('Finished Data Ingestion component. Train, test and validation entire sets obtained (artifacts paths).')

            logging.info('Data Transformation component started.')
            
            data_transformation = DataTransformation()
            train_prepared, test_prepared, val_prepared, _ = data_transformation.apply_data_transformation(train_path, test_path, val_path)

            logging.info('Finished Data Transformation component. Train, test and validation entire prepared sets obtained (artifacts).')

            logging.info('Model Trainer component started.')
            
            model_trainer = ModelTrainer()

            MAE, RMSE, R2 = model_trainer.apply_model_trainer(train_prepared, test_prepared, val_prepared)

            print(f'MAE: {MAE}')
            print(f'RMSE: {RMSE}')
            print(f'R2: {R2}')

            logging.info('Finished Model Trainer component. Final best model obtained (artifacts).')

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':

    # Apply full train pipeline using the components above.
    train_pipeline = TrainPipeline()
    train_pipeline.train()