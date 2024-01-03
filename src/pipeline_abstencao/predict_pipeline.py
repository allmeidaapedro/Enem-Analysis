'''
This script aims to create the predict pipeline for a simple web application which will be interacting with the pkl files, such that we can make predictions by giving values of input features. 
'''

# Debugging and verbose.
import sys
from src.logger import logging
from src.exception import CustomException
from src.artifacts_utils import load_object
from src.data_cleaning import data_cleaning

# Data manipulation.
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)

# File handling.
import os


class PredictPipeline:
    '''
    Class for making predictions using a trained model and preprocessor.

    This class provides a pipeline for making predictions on new instances using a trained machine learning model and
    a preprocessor. It loads the model and preprocessor from files, cleans and preprocesses the input features, and makes predictions.

    Methods:
        predict(features):
            Make predictions on new instances using the loaded model and preprocessor.

    Example:
        pipeline = PredictPipeline()
        new_features = [...]
        prediction = pipeline.predict(new_features)

    Note:
        This class assumes the availability of the load_object function.
    '''

    
    def __init__(self) -> None:
        '''
        Initializes a PredictPipeline instance.

        Initializes the instance. No specific setup is required in the constructor.
        '''
        pass


    def predict(self, features):
        '''
        Make predictions on new instances using the loaded model and preprocessor.

        Args:
            features (pd.DataFrame): Input features for which predictions will be made.

        Returns:
            predictions: Predicted labels for the input features.

        Raises:
            CustomException: If an exception occurs during the prediction process.
        '''
        try:
            model_path = os.path.join('artifacts', 'model_abstencao.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor_abstencao.pkl')
            
            logging.info('Load model and preprocessor objects.')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info('Model and preprocessor succesfully loaded.')

            logging.info('Cleaning the input data.')

            clean_input_data = data_cleaning(features, absence=True, raw_data=False)

            logging.info('Preprocessing the input data.')

            # Imputing 'Não sei' (null value) category with train mode for escolaridade_pai and escolaridade_mae. The mode is 'Ensino médio completo'.
            train_path = os.path.join('artifacts', 'train_abstencao.csv')
            train = pd.read_csv(train_path)

            esc_mae_mode = train['escolaridade_mae'].mode()[0]
            esc_pai_mode = train['escolaridade_pai'].mode()[0]

            if (clean_input_data['escolaridade_mae'] == 'Não sei').any():
                clean_input_data['escolaridade_mae'] = clean_input_data['escolaridade_mae'].replace(to_replace={'Não sei': esc_mae_mode})
            if (clean_input_data['escolaridade_pai'] == 'Não sei').any():
                clean_input_data['escolaridade_pai'] = clean_input_data['escolaridade_pai'].replace(to_replace={'Não sei': esc_pai_mode})

            prepared_input_data = preprocessor.transform(clean_input_data)

            logging.info('Input data prepared for prediction.')

            logging.info('Predicting.')
            
            # Predicting candidate's absence probability.
            predicted_proba = model.predict_proba(prepared_input_data)[:, 1][0]

            # Prediction output (candidate's absence probability on enem).
            prediction = f'Probabilidade de Abstenção = {round(predicted_proba * 100, 2)}%'

            print(prediction)

            logging.info('Prediction successfully made.')

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
        

class InputData:
    '''
    Class for handling input data for predictions.

    This class provides a structured representation for input data that is meant to be used for making predictions.
    It maps input variables from HTML inputs to class attributes and provides a method to convert the input data into
    a DataFrame format suitable for making predictions.

    Attributes:
        faixa_etaria (str)              
        status_conclusao_ensino_medio (str)                      
        escolaridade_pai (str)             
        escolaridade_mae (str)             
        numero_pessoas_em_casa (int)              
        renda_familiar_mensal (str)             
        escola (str)               
        uf_prova (str)              
        possui_celular_em_casa (str)    
        possui_computador_em_casa (str)      
        sexo (str)         
        treineiro (str)             
        lingua (str)       
        acesso_internet_em_casa (str)                 

    Methods:
        get_input_data_df():
            Convert the mapped input data into a DataFrame for predictions.

    Note:
        This class assumes the availability of the pandas library and defines the CustomException class.
    '''

    def __init__(self,
                 faixa_etaria: str,
                 status_conclusao_ensino_medio: str,
                 escolaridade_pai: str,
                 escolaridade_mae: str,
                 numero_pessoas_em_casa: int,
                 renda_familiar_mensal: str,
                 escola: str,
                 uf_prova: str,
                 possui_celular_em_casa: str,
                 possui_computador_em_casa: str,
                 sexo: str,
                 treineiro: str,
                 lingua: str,
                 acesso_internet_em_casa: str
                 ) -> None:
        '''
        Initialize an InputData instance with mapped input data.

        Args:
            faixa_etaria (str)              
            status_conclusao_ensino_medio (str)                      
            escolaridade_pai (str)             
            escolaridade_mae (str)             
            numero_pessoas_em_casa (int)              
            renda_familiar_mensal (str)             
            escola (str)               
            uf_prova (str)              
            possui_celular_em_casa (str)    
            possui_computador_em_casa (str)      
            sexo (str)         
            treineiro (str)             
            lingua (str)       
            acesso_internet_em_casa (str)  
        '''
        
        # Map variables from html inputs.
        self.faixa_etaria = faixa_etaria
        self.status_conclusao_ensino_medio = status_conclusao_ensino_medio
        self.escolaridade_pai = escolaridade_pai
        self.escolaridade_mae = escolaridade_mae
        self.numero_pessoas_em_casa = numero_pessoas_em_casa
        self.renda_familiar_mensal = renda_familiar_mensal
        self.escola = escola
        self.uf_prova = uf_prova
        self.possui_celular_em_casa = possui_celular_em_casa
        self.possui_computador_em_casa = possui_computador_em_casa
        self.sexo = sexo
        self.treineiro = treineiro
        self.lingua = lingua
        self.acesso_internet_em_casa = acesso_internet_em_casa

    def get_input_data_df(self):
        '''
        Convert the mapped input data into a DataFrame for predictions.

        Returns:
            input_data_df (DataFrame): DataFrame containing the mapped input data.

        Raises:
            CustomException: If an exception occurs during the process.
        '''
        try:
            input_data_dict = dict()

            # Map the variables to the form of a dataframe for being used in predictions.
            
            input_data_dict['faixa_etaria'] = [self.faixa_etaria]
            input_data_dict['status_conclusao_ensino_medio'] = [self.status_conclusao_ensino_medio]
            input_data_dict['escolaridade_pai'] = [self.escolaridade_pai]
            input_data_dict['escolaridade_mae'] = [self.escolaridade_mae]
            input_data_dict['numero_pessoas_em_casa'] = [self.numero_pessoas_em_casa]
            input_data_dict['renda_familiar_mensal'] = [self.renda_familiar_mensal]
            input_data_dict['escola'] = [self.escola]
            input_data_dict['uf_prova'] = [self.uf_prova]
            input_data_dict['possui_celular_em_casa'] = [self.possui_celular_em_casa]
            input_data_dict['possui_computador_em_casa'] = [self.possui_computador_em_casa]
            input_data_dict['sexo'] = [self.sexo]
            input_data_dict['treineiro'] = [self.treineiro]
            input_data_dict['lingua'] = [self.lingua]
            input_data_dict['acesso_internet_em_casa'] = [self.acesso_internet_em_casa]

            input_data_df = pd.DataFrame(input_data_dict)

            return input_data_df
        
        except Exception as e:
            raise CustomException(e, sys)