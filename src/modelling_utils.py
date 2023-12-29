'''
This script aims to provide functions that will turn the modelling process easier
'''

'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modelling.
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import math

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

# Definições de cores -> todas estão numa escala de mais escura para mais clara.
CINZA1, CINZA2, CINZA3 = '#231F20', '#414040', '#555655'
CINZA4, CINZA5, CINZA6 = '#646369', '#76787B', '#828282'
CINZA7, CINZA8, CINZA9 = '#929497', '#A6A6A5', '#BFBEBE'
AZUL1, AZUL2, AZUL3, AZUL4 = '#174A7E', '#4A81BF', '#94B2D7', '#94AFC5'
VERMELHO1, VERMELHO2, VERMELHO3, VERMELHO4, VERMELHO5 = '#DB0527', '#E23652', '#ED8293', '#F4B4BE', '#FBE6E9'
VERDE1, VERDE2 = '#0C8040', '#9ABB59'
LARANJA1 = '#F79747'
AMARELO1, AMARELO2, AMARELO3, AMARELO4, AMARELO5 = '#FFC700', '#FFCC19', '#FFEB51', '#FFE37F', '#FFEEB2'
BRANCO = '#FFFFFF'


def evaluate_models_cv(models, X_train, y_train, n_folds=5):
    '''
    Evaluate multiple machine learning models using stratified k-fold cross-validation (the stratified k-fold is useful for dealing with target imbalancement).

    This function evaluates a dictionary of machine learning models by training each model on the provided training data
    and evaluating their performance using stratified k-fold cross-validation. The evaluation metric used is ROC-AUC score.

    Args:
        models (dict): A dictionary where the keys are model names and the values are instantiated machine learning model objects.
        X_train (array-like): The training feature data.
        y_train (array-like): The corresponding target labels for the training data.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each model, including their average validation scores
                  and training scores.

    Raises:
        CustomException: If an error occurs while evaluating the models.

    '''


    try:
        # Dictionaries with validation and training scores of each model for plotting further.
        models_val_scores = dict()
        models_train_scores = dict()

        for model in models:
            # Getting the model object from the key with his name.
            model_instance = models[model]

            # Measuring training time.
            start_time = time.time()
            
            # Fitting the model to the training data.
            model_instance.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time

            # Make predictions on training data and evaluate them.
            y_train_pred = model_instance.predict(X_train)
            train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))

            # Evaluate the model using k-fold cross validation, obtaining a robust measurement of its performance on unseen data.
            val_scores = cross_val_score(model_instance, X_train, y_train, scoring='neg_mean_squared_error', cv=n_folds)
            avg_val_score = np.sqrt(-1 * val_scores.mean())
            val_score_std = np.sqrt((-1 * val_scores)).std()

            # Adding the model scores to the validation and training scores dictionaries.
            models_val_scores[model] = avg_val_score
            models_train_scores[model] = train_score

            # Printing the results.
            print(f'{model} results: ')
            print('-'*50)
            print(f'Training score: {train_score}')
            print(f'Average validation score: {avg_val_score}')
            print(f'Standard deviation: {val_score_std}')
            print(f'Training time: {round(training_time, 5)} seconds')
            print()


        # Plotting the results.
        print('Plotting the results: ')

        # Converting scores to a dataframe
        val_df = pd.DataFrame(list(models_val_scores.items()), columns=['Model', 'Average Val Score'])
        train_df = pd.DataFrame(list(models_train_scores.items()), columns=['Model', 'Train Score'])
        eval_df = val_df.merge(train_df, on='Model')

        # Sorting the dataframe by the best RMSE.
        eval_df  = eval_df.sort_values(['Average Val Score'], ascending=True).reset_index(drop=True)

        # Plotting each model and their train and validation (average) scores.
        fig, ax = plt.subplots(figsize=(20, 6))
        width = 0.35

        x = np.arange(len(eval_df['Model']))
        y = np.arange(len(eval_df['Train Score']))

        val_bars = ax.bar(x - width/2, eval_df['Average Val Score'], width, label='Average Validation Score', color=LARANJA1)
        train_bars = ax.bar(x + width/2, eval_df['Train Score'], width, label='Train Score', color=AZUL1)

        ax.set_xlabel('Model', color=CINZA1, labelpad=20, fontsize=10.8)
        ax.set_ylabel('RMSE Score', color=CINZA1, labelpad=20, fontsize=10.8)
        ax.set_title("Models' Performances", fontweight='bold', fontsize=12, pad=30, color=CINZA1)
        ax.set_xticks(x, eval_df['Model'], rotation=0, color=CINZA1, fontsize=10.8)
        ax.tick_params(axis='y', color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)

        # Add scores on top of each bar
        for bar in val_bars + train_bars:
            height = bar.get_height()
            plt.annotate('{}'.format(round(height, 2)),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', color=CINZA1)

        ax.legend(loc='upper left')

        return eval_df
    
    except Exception as e:
        raise(CustomException(e, sys))


def compare_actual_predicted(y_true, y_pred):
    '''
    Compares actual and predicted values and calculates the residuals.

    Args:
    y_true : The true target values.
    y_pred : The predicted target values.

    Returns:
    pandas.DataFrame: A dataframe containing the actual, predicted, and residual values.

    Raises:
    CustomException: An error occurred during the comparison process.
    '''
    try:
        actual_pred_df = pd.DataFrame({'Actual': np.round(y_true, 2),
                                    'Predicted': np.round(y_pred, 2), 
                                    'Residual': np.round(np.abs(y_pred - y_true), 2)})
        return actual_pred_df
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_regressor(y_true, y_pred, y_train, model_name):
    '''
    Evaluates a regression model based on various metrics and plots.

    Args:
    y_true : The true target values.
    y_pred : The predicted target values.
    y_train : The actual target values from the training set.
    model_name (str): The name of the regression model.

    Returns:
    pandas.DataFrame: A dataframe containing the evaluation metrics.

    Raises:
    CustomException: An error occurred during the evaluation process.
    '''
    try:
        mae = round(mean_absolute_error(y_true, y_pred), 4)
        mse = round(mean_squared_error(y_true, y_pred), 4)
        rmse = round(np.sqrt(mse), 4)
        r2 = round(r2_score(y_true, y_pred), 4)
        mape = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 4)
        
        # Metrics
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Mean Absolute Percentage Error (MAPE): {mape}')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'R-Squared (R2): {r2}')
        
        # Obtaining a dataframe of the metrics.
        df_results = pd.DataFrame({'Model': model_name, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2}, index=['Results'])

        # Residual Plots
        
        # Analysing the results
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title('Valores Verdadeiros vs Valores Preditos', fontweight='bold', fontsize=12, color=CINZA1, pad=20)
        ax.plot([y_train.min(),y_train.max()],[y_train.min(),y_train.max()], color=LARANJA1, linestyle='--')
        ax.scatter(y_true, y_pred, color=AZUL1)
        ax.set_xlabel('Verdadeiro', fontsize=10.8, color=CINZA1, labelpad=20)
        ax.set_ylabel('Predito', fontsize=10.8, color=CINZA1, labelpad=20)
        ax.tick_params(axis='both', color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        
        # Distribution of the residuals
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.distplot((y_true - y_pred), ax=ax, color=AZUL1)
        ax.set_title('Distribuição dos Resíduos', fontweight='bold', fontsize=12, color=CINZA1, pad=20)
        ax.set_xlabel('Resíduos', fontsize=10.8, color=CINZA1, labelpad=20)
        ax.set_ylabel('Frequência', fontsize=10.8, color=CINZA1, labelpad=20)
        ax.tick_params(axis='both', color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        
        return df_results

    except Exception as e:
        raise CustomException(e, sys)


def linear_interpretation_df(variables, coefficients):
    '''
    Construct a DataFrame for the interpretation of coefficients from a linear regression.

    Parameters:
    - variables (list): List of variable names.
    - coefficients (list): List of corresponding coefficients.

    Returns:
    pandas.DataFrame: DataFrame with variables as the index, and columns for coefficient and direction of the relationship.

    Example:
    ```python
    variables = ['Variable1', 'Variable2', 'Variable3']
    coefficients = [0.5, -0.8, 0.2]
    result_df = linear_interpretation_df(variables, coefficients)
    ```

    The resulting DataFrame will have variables as the index, with columns for the coefficient and the direction of the relationship (positive, negative, or irrelevant).

    Raises:
    - CustomException: If an exception occurs during the DataFrame creation.

    Note:
    - The direction of the relationship is categorized as 'Positive' if the coefficient is greater than 0, 'Negative' if less than 0, and 'Irrelevant' if 0.

    '''
    try:
        coef_df = pd.DataFrame({'Variável': variables, 
                                'Coeficiente': coefficients})
        coef_df['Correlação'] = coef_df['Coeficiente'].apply(lambda x: 'Positiva' if x > 0 else 'Negativa' if x < 0 else 'Irrelevante')
        coef_df = coef_df.reindex(coef_df['Coeficiente'].abs().sort_values(ascending=False).index)
        coef_df.set_index('Variável', inplace=True)
        return coef_df
    except Exception as e:
        raise CustomException(e, sys)
