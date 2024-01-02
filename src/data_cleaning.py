'''
This script aims to perform initial data cleaning and feature engineering on the performance and absent datasets required for data splitting.
'''

'''
Importing libraries
'''

# Debugging, verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# Data manipulation.
import pandas as pd
import numpy as np

# Warnings.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def data_cleaning(df, abstencao=False, input=False):
    '''
    Perform data cleaning and feature engineering on a DataFrame for performance and abstencao modelling tasks.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the raw data.
    - input (bool): Wheter we should clean just input features. Default is False.

    Returns:
    pd.DataFrame: A cleaned and feature-engineered DataFrame.

    Raises:
    CustomException: An exception raised during the execution of the function.

    This function performs the following tasks:
    1. Maps categorical variables to binary or ordinal values.
    2. Performs feature engineering to create new features like 'nota_geral', 'regiao', 'acesso_tecnologia', and 'renda_por_pessoa'.
    3. Bins the 'numero_pessoas_em_casa' variable into categories.
    4. Removes observations with zero grades if desempenho modelling is being performed.
    5. Drops irrelevant variables to avoid data leakage.
    6. Mixes categories based on observed proportions and absent rates for 'escolaridade_mae', 'escolaridade_pai', 'renda_familiar_mensal', and 'faixa_etaria'.

    Note:
    - The input DataFrame 'df' should have specific columns for this function to work correctly.

    Example:
    ```python
    import pandas as pd

    # Load raw data
    raw_data = pd.read_csv('raw_data.csv')

    # Perform cleaning and feature engineering
    cleaned_data = performance_cleaning(raw_data)
    ```
    '''
    try:
        clean_df = df.copy()
        
        # Feature engineering.

        # Mapping variables to binary or ordinal.
        clean_df['sexo'] = clean_df['sexo'].replace(to_replace={'M': 1, 'F': 0}).astype('int8')
        clean_df['lingua'] = clean_df['lingua'].replace(to_replace={'Inglês': 1, 'Espanhol': 0}).astype('int8')
        clean_df['treineiro'] = clean_df['treineiro'].replace(to_replace={'Sim': 1, 'Não': 0}).astype('int8')
        clean_df['acesso_internet_em_casa'] = clean_df['acesso_internet_em_casa'].replace(to_replace={'Sim': 1, 'Não': 0}).astype('int8')
        clean_df['possui_celular_em_casa'] = clean_df['possui_celular_em_casa'].replace(to_replace={'Dois ou mais': 2, 'Um': 1, 'Não': 0}).astype('int8')
        clean_df['possui_computador_em_casa'] = clean_df['possui_computador_em_casa'].replace(to_replace={'Dois ou mais': 2, 'Um': 1, 'Não': 0}).astype('int8')

        # Creating a feature indicating the region where the students' exams were applied.
        def define_regions(x):
            if x in set(['RS', 'SC', 'PR']):
                return 'Sul'
            elif x in set(['SP', 'RJ', 'MG', 'ES']):
                return 'Sudeste'
            elif x in set(['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE']):
                return 'Nordeste'
            elif x in set(['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO']):
                return 'Norte'
            else:
                return 'Centro-Oeste'
                
        clean_df['regiao'] = clean_df['uf_prova'].apply(define_regions)

        # Creating a variable that indicates the student's access to technology.
        clean_df['acesso_tecnologia'] = clean_df['acesso_internet_em_casa'] + clean_df['possui_celular_em_casa'] + clean_df['possui_computador_em_casa']

        # Creating a variable that indicates the income per people at home.
        # Mapping income categories to numeric values
        income_mapping = {
            'Até R$ 1.212,00': 1212,
            'Nenhuma Renda': 0,
            'R$ 1.818,01 - R$ 3.030,00': (1818 + 3030) / 2,
            'R$ 1.212,01 - R$ 1.818,00': (1212 + 1818) / 2,
            'R$ 3.030,01 - R$ 4.848,00': (3030 + 4848) / 2,
            'R$ 4.848,01 - R$ 7.272,00': (4848 + 7272) / 2,
            'R$ 7.272,01 - R$ 10.908,00': (7272 + 10908) / 2,
            'Acima de R$ 24.240,00': 24240,
            'R$ 18.180,01 - R$ 24.240,00': (18180 + 24240) / 2,
            'R$ 10.908,01 - R$ 18.180,00': (10908 + 18180) / 2
        }

        # Map income categories to numeric values
        clean_df['renda_numerica'] = clean_df['renda_familiar_mensal'].map(income_mapping).astype('int32')

        # Calculate income per person
        # Converting selected input 'numero_pessoas_em_casa' to int.
        clean_df['numero_pessoas_em_casa'] = clean_df['numero_pessoas_em_casa'].astype('int8')
        clean_df['renda_por_pessoa'] = (clean_df['renda_numerica'] / clean_df['numero_pessoas_em_casa']).astype('float32')

        # Binning number of people at home variable.
        clean_df['numero_pessoas_em_casa'] = pd.cut(clean_df['numero_pessoas_em_casa'],
                                                bins=[1, 3, 5, 10, 20],
                                                labels=['1 a 3', '4 a 5', '6 a 10', '11 a 20'], include_lowest=True)
        
        # Mixing categories based on observed proportions and absent rates.
        clean_df['escolaridade_mae'] = clean_df['escolaridade_mae'].replace(to_replace={'Pós-graduação': 'Ensino superior completo'})
        clean_df['escolaridade_pai'] = clean_df['escolaridade_pai'].replace(to_replace={'Pós-graduação': 'Ensino superior completo'})

        # Combining similar low proportion income categories.
        clean_df['renda_familiar_mensal'] = clean_df['renda_familiar_mensal'].replace(to_replace={
            'Até R$ 1.212,00': 'Renda baixa',
            'R$ 1.212,01 - R$ 1.818,00': 'Renda baixa',
            'R$ 1.818,01 - R$ 3.030,00': 'Renda média baixa',
            'R$ 3.030,01 - R$ 4.848,00': 'Renda média baixa',
            'R$ 4.848,01 - R$ 7.272,00': 'Renda média alta',
            'R$ 7.272,01 - R$ 10.908,00': 'Renda média alta', 
            'R$ 10.908,01 - R$ 18.180,00': 'Renda alta',
            'R$ 18.180,01 - R$ 24.240,00': 'Renda alta',
            'Acima de R$ 24.240,00': 'Renda alta'
        })

        if abstencao:
            # Combining similar low proportion age categories.
            clean_df['faixa_etaria'] = clean_df['faixa_etaria'].replace(to_replace={'Adulto jovem (25-35)': 'Adulto (25-45)', 
                                                                        'Adulto de meia idade (36-45)': 'Adulto (25-45)',
                                                                        'Meia idade (46-55)': 'Meia idade a idoso (46+)',
                                                                        'Pré aposentadoria (56-65)': 'Meia idade a idoso (46+)',
                                                                        'Idoso (> 66)': 'Meia idade a idoso (46+)'})
        else:
            # Combining similar low proportion age categories.
            clean_df['faixa_etaria'] = clean_df['faixa_etaria'].replace(to_replace={'Adulto de meia idade (36-45)': 'Adulto a meia idade (36-55)', 'Meia idade (46-55)': 'Adulto a meia idade (36-55)', 'Pré aposentadoria (56-65)': 'Pré aposentadoria a idoso (> 56)', 'Idoso (> 66)': 'Pré aposentadoria a idoso (> 56)'})

        if not input:
            if abstencao:
                # Creating our target, a feature indicating wheter the student was absent in at least one day of the exam (1).
                clean_df['abstencao'] = (clean_df['presenca_lc'] == 'Ausente') | \
                                    (clean_df['presenca_ch'] == 'Ausente') | \
                                    (clean_df['presenca_cn'] == 'Ausente') | \
                                    (clean_df['presenca_mt'] == 'Ausente') 
                clean_df['abstencao'] = clean_df['abstencao'].astype('int8')
            else:
                # Creating a feature indicating the students' average grade (TARGET).
                clean_df['nota_geral'] = (clean_df['nota_lc'] + clean_df['nota_ch'] + clean_df['nota_cn'] + clean_df['nota_mt'] + clean_df['nota_redacao']) / 5

                # Removing zero grade observations.
                clean_df = clean_df.loc[~(clean_df['nota_geral'] == 0)]

            # Dropping irrelevant variables or variables which would lead to data leakage.
            to_drop = ['municipio_prova', 'presenca_cn', 'presenca_ch', 'presenca_lc', 
            'presenca_mt', 'nota_cn', 'nota_ch', 'nota_lc', 
            'nota_mt', 'nota_comp1', 'nota_comp2', 'nota_comp3', 
            'nota_comp4', 'nota_comp5', 'nota_redacao', 'estado_civil', 
            'uf_prova', 'renda_numerica']

            clean_df = clean_df.drop(columns=to_drop)

            return clean_df

        to_drop = ['renda_numerica']
        clean_df = clean_df.drop(columns=to_drop)

        return clean_df
    
    except Exception as e:
         raise CustomException(e, sys)


