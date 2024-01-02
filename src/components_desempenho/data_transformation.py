'''
This script aims to apply data cleaning and preprocessing to the data.
'''

# Debugging and verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# File handling.
import os

# Data manipulation.
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Preprocessing.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Utils.
from src.artifacts_utils import save_object


@dataclass
class DataTransformationConfig:
    '''
    Configuration class for data transformation.

    This data class holds configuration parameters related to data transformation. It includes attributes such as
    `preprocessor_file_path` that specifies the default path to save the preprocessor object file.

    Attributes:
        preprocessor_file_path (str): The default file path for saving the preprocessor object. By default, it is set
                                     to the 'artifacts' directory with the filename 'preprocessor_desempenho.pkl'.

    Example:
        config = DataTransformationConfig()
        print(config.preprocessor_file_path)  # Output: 'artifacts/preprocessor_desempenho.pkl'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    preprocessor_file_path = os.path.join('artifacts', 'preprocessor_desempenho.pkl')


class DataTransformation:
    '''
    Data transformation class for preprocessing and transformation of train, test and validation sets.

    This class handles the preprocessing and cleaning of datasets, including
    categorical encoding and feature scaling.

    :ivar data_transformation_config: Configuration instance for data transformation.
    :type data_transformation_config: DataTransformationConfig
    '''
    def __init__(self) -> None:
        '''
        Initialize the DataTransformation instance with a DataTransformationConfig.
        '''
        self.data_transformation_config = DataTransformationConfig()


    def get_preprocessor(self):
        '''
        Get a preprocessor for data transformation.

        This method sets up pipelines for ordinal encoding, one-hot encoding,
        and scaling of features.

        :return: Preprocessor object for data transformation.
        :rtype: ColumnTransformer
        :raises CustomException: If an exception occurs during the preprocessing setup.
        '''

        try:
            ordinal_features = ['faixa_etaria', 'status_conclusao_ensino_medio', 
                                'escolaridade_pai', 'escolaridade_mae', 
                                'numero_pessoas_em_casa', 'renda_familiar_mensal']
            nominal_features = ['escola', 'regiao']
            scale_features = ['possui_celular_em_casa', 'possui_computador_em_casa', 
                             'acesso_tecnologia', 'renda_por_pessoa']

            logging.info('Constructing the preprocessor.')
            logging.info(f'Ordinal features (ordinal encoding and standard scaling): {ordinal_features}')
            logging.info(f'Nominal features (one-hot encoding): {nominal_features}')
            logging.info(f'Features to scale (standard scaling): {scale_features}')

            # Ordinal variables categories orders.
            age_categories_ordered = ['Adolescente (< 18)', 'Jovem adulto (18-24)', 'Adulto jovem (25-35)', 'Adulto a meia idade (36-55)', 'Pré aposentadoria a idoso (> 56)']
            
            status_categories_ordered = ['Não concluído', 'Cursando', 'Último ano', 'Concluído']
            
            educational_level_categories_ordered = ['Nunca estudou', 'Ensino fundamental incompleto', 'Ensino fundamental completo', 'Ensino médio completo', 'Ensino superior completo']

            people_home_categories_ordered = ['1 a 3', '4 a 5', '6 a 10', '11 a 20']

            income_categories_ordered = ['Nenhuma Renda', 'Renda baixa', 'Renda média baixa', 'Renda média alta', 'Renda alta']

            # Ordinal encoding pipeline.
            ordinal_pipe = Pipeline([
                ('ordinal_encoder', ColumnTransformer(
                    transformers=[
                        ('age', OrdinalEncoder(categories=[age_categories_ordered]), ['faixa_etaria']),
                        ('status', OrdinalEncoder(categories=[status_categories_ordered]), ['status_conclusao_ensino_medio']),
                        ('dad', OrdinalEncoder(categories=[educational_level_categories_ordered]), ['escolaridade_pai']),
                        ('mom', OrdinalEncoder(categories=[educational_level_categories_ordered]), ['escolaridade_mae']),
                        ('people', OrdinalEncoder(categories=[people_home_categories_ordered]), ['numero_pessoas_em_casa']),
                        ('income', OrdinalEncoder(categories=[income_categories_ordered]), ['renda_familiar_mensal'])
                    ]
                )),
                ('std_scaler', StandardScaler())
            ])

            # One hot encoding pipeline.
            nominal_pipe = Pipeline([
                ('one_hot_encoder', OneHotEncoder())
            ])

            # Standard Scaler (on the mapped ordinal features) pipeline.
            scaling_pipe = Pipeline([
                ('std_scaler', StandardScaler())
            ])

            # Column transformer.
            preprocessor = ColumnTransformer(
                transformers=[
                    ('ordinal_encoding', ordinal_pipe, ordinal_features),
                    ('one_hot_encoding', nominal_pipe, nominal_features),
                    ('std_scaling', scaling_pipe, scale_features)
                ], remainder='passthrough'
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def apply_data_transformation(self, train_path, test_path, val_path):
        '''
        Apply data transformation process.

        Reads, preprocesses, and transforms training, testing and validation datasets.

        :param train_path: Path to the training dataset CSV file.
        :param test_path: Path to the test dataset CSV file.
        :param val_path: Path to the validation dataset CSV file.
        :return: Prepared training, testing and validation datasets and the preprocessor file path.
        :rtype: tuple
        :raises CustomException: If an exception occurs during the data transformation process.
        '''
        
        try:

            logging.info('Reading train, test and validation sets.')

            # Obtaining train, test and validation entire sets from artifacts.
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            val = pd.read_csv(val_path)

            logging.info('Obtaining preprocessor object.')

            preprocessor = self.get_preprocessor()

            # Getting train, test and validation predictor and target sets.
            X_train = train.drop(columns=['nota_geral'])
            y_train = train['nota_geral'].copy()

            X_test = test.drop(columns=['nota_geral'])
            y_test = test['nota_geral'].copy()

            X_val = val.drop(columns=['nota_geral'])
            y_val = val['nota_geral'].copy()

            logging.info('Preprocessing train, test and validation sets.')

            # Imputing 'Não sei' with the mode.
            esc_mae_mode = X_train['escolaridade_mae'].mode()[0]
            esc_pai_mode = X_train['escolaridade_pai'].mode()[0]

            X_train['escolaridade_mae'] = X_train['escolaridade_mae'].replace(to_replace={'Não sei': esc_mae_mode})
            X_train['escolaridade_pai'] = X_train['escolaridade_pai'].replace(to_replace={'Não sei': esc_pai_mode})

            X_val['escolaridade_mae'] = X_val['escolaridade_mae'].replace(to_replace={'Não sei': esc_mae_mode})
            X_val['escolaridade_pai'] = X_val['escolaridade_pai'].replace(to_replace={'Não sei': esc_pai_mode})

            X_test['escolaridade_mae'] = X_test['escolaridade_mae'].replace(to_replace={'Não sei': esc_mae_mode})
            X_test['escolaridade_pai'] = X_test['escolaridade_pai'].replace(to_replace={'Não sei': esc_pai_mode})

            X_train_prepared = preprocessor.fit_transform(X_train, y_train)
            X_test_prepared = preprocessor.transform(X_test)
            X_val_prepared = preprocessor.transform(X_val)

            # Getting final train, test and validation entire prepared sets.
            # Obtaining DataFrame format to concatenate train and validation sets later in model_trainer.
            cols = ['faixa_etaria', 'status_conclusao_ensino_medio', 'escolaridade_pai','escolaridade_mae', 'numero_pessoas_em_casa', 'renda_familiar_mensal','nao_respondeu_escola', 'escola_privada', 'escola_publica','centro_oeste', 'nordeste', 'norte', 'sudeste', 'sul', 'possui_celular_em_casa', 'possui_computador_em_casa', 'acesso_tecnologia', 'renda_por_pessoa', 'sexo', 'treineiro', 'lingua', 'acesso_internet_em_casa', 'nota_geral']

            X_train_prepared_df = pd.DataFrame(X_train_prepared)
            X_test_prepared_df = pd.DataFrame(X_test_prepared)
            X_val_prepared_df = pd.DataFrame(X_val_prepared)

            train_prepared = pd.concat([X_train_prepared_df, y_train], axis=1)
            test_prepared = pd.concat([X_test_prepared_df, y_test], axis=1)
            val_prepared = pd.concat([X_val_prepared_df, y_val], axis=1)

            train_prepared.columns = cols
            test_prepared.columns = cols
            val_prepared.columns = cols

            logging.info('Entire train, test and validation sets prepared.')

            logging.info('Save preprocessing object.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                object=preprocessor
            )
        
            return train_prepared, test_prepared, val_prepared, self.data_transformation_config.preprocessor_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
        