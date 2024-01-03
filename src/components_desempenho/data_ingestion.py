'''
This script aims to read the dataset from the data source and split it into train and test sets. These steps results are necessary for data transformation component.
'''

'''
Importing libraries
'''


# Debugging, verbose.
import sys
from src.exception import CustomException
from src.logger import logging
from src.data_cleaning import data_cleaning

# File handling.
import os

# Data manipulation.
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Warnings.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class DataIngestionConfig:
    '''
    Configuration class for data ingestion.

    This data class holds configuration parameters related to data ingestion. It includes attributes such as
    `train_data_path`, `test_data_path`, `val_data_path` and `raw_data_path` that specify the default paths for train, test, validation and raw data files respectively.

    Attributes:
        train_data_path (str): The default file path for the training data. By default, it is set to the 'artifacts'
                              directory with the filename 'train_desempenho.csv'.
        test_data_path (str): The default file path for the test data. By default, it is set to the 'artifacts'
                             directory with the filename 'test_desempenho.csv'.
        val_data_path (str): The default file path for the validation data. By default, it is set to the 'artifacts'
                             directory with the filename 'val_desempenho.csv'.
        raw_data_path (str): The default file path for the raw data. By default, it is set to the 'artifacts'
                            directory with the filename 'data_desempenho.csv'.

    Example:
        config = DataIngestionConfig()
        print(config.train_data_path)  # Output: 'artifacts/train_desempenho.csv'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    train_data_path = os.path.join('artifacts', 'train_desempenho.csv')
    test_data_path = os.path.join('artifacts', 'test_desempenho.csv')
    val_data_path = os.path.join('artifacts', 'val_desempenho.csv')
    raw_data_path = os.path.join('artifacts', 'data_desempenho.csv')


class DataIngestion:
    '''
    Data ingestion class for preparing and splitting datasets.

    This class handles the data ingestion process, including reading, splitting,
    and saving the raw data, training data, test data and validation data.

    :ivar ingestion_config: Configuration instance for data ingestion.
    :type ingestion_config: DataIngestionConfig
    '''

    def __init__(self) -> None:
        '''
        Initialize the DataIngestion instance with a DataIngestionConfig.
        '''
        self.ingestion_config = DataIngestionConfig()

    def apply_data_ingestion(self):
        '''
        Apply data ingestion process.

        Reads the dataset, performs train-test-validation split,
        and saves raw data, training data, test data and validation data.

        :return: Paths to the saved training data, test data and validation data files.
        :rtype: tuple
        :raises CustomException: If an exception occurs during the data ingestion process.
        '''
        
        try:
            logging.info('Reading the dataset as a Pandas DataFrame and saving it as a csv.')

            path = 'D:\\MLProjects\\EnemAnalysis\\input\data\\clean_df.parquet'
            df = pd.read_parquet(path)

            # Data that is being used for performance modelling consists only of the candidates who were present in both days of the exam.
            df = df.loc[(df['presenca_lc'] == 'Presente') 
                        & (df['presenca_ch'] == 'Presente') 
                        & (df['presenca_cn'] == 'Presente') 
                        & (df['presenca_mt'] == 'Presente')].reset_index(drop=True)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Cleaning the data.')
            
            clean_df = data_cleaning(df, absence=False, raw_data=True)

            logging.info('Obtaining X and y.')

            X = clean_df.drop(columns=['nota_geral'])
            y = clean_df['nota_geral'].copy()

            logging.info('Train-test-validation split.')

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
            X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # Getting back train, test and validation entire sets.
            train = pd.concat([X_train, y_train], axis=1)
            test = pd.concat([X_test, y_test], axis=1)
            val = pd.concat([X_val, y_val], axis=1)

            logging.info('Saving train, test and validation sets into a csv.')
            
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            val.to_csv(self.ingestion_config.val_data_path, index=False, header=True)

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path, self.ingestion_config.val_data_path
        
        except Exception as e:
            raise CustomException(e, sys)
        