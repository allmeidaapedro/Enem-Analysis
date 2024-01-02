'''
This script aims to build a simple web application using Flask. The web application will interact with the ML model artifacts such that we can make predictions by giving input features values.
'''

'''
Importing the libraries.
'''

# Web app.
from flask import Flask, request, render_template

# Data manipulation.
import numpy as np
import pandas as pd

# File handling.
import os

# Predictions.
from src.pipeline_abstencao.predict_pipeline import InputData, PredictPipeline


application = Flask(__name__)


app = application


# Route for the home page.

@app.route('/')
def index():
    '''
    Route handler for the home page.

    This function handles the GET request for the home page. It renders the 'index_abstencao.html' template, which serves as the
    homepage for the enem candidate's absence probability prediction web application.

    :return: The rendered home page.
    :rtype: str
    '''
    return render_template('index_abstencao.html')


# Route for prediction page.

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    '''
    Route handler for predicting credit risk.

    This function handles the POST request for predicting enem candidate's absence probability based on input data. If the request is a GET request,
    the function renders the 'home_abstencao.html' template. If the request is a POST request, it collects input data from the form,
    processes it to make a prediction, and returns the prediction result.

    :return: The prediction result.
    :rtype: str
    '''

    if request.method == 'GET':
        return render_template('home_abstencao.html')
    else:
        input_data = InputData(
            faixa_etaria=request.form.get('faixa_etaria'),
            status_conclusao_ensino_medio=request.form.get('status_conclusao_ensino_medio'),
            escolaridade_pai=request.form.get('escolaridade_pai'),
            escolaridade_mae=request.form.get('escolaridade_mae'),
            numero_pessoas_em_casa=request.form.get('numero_pessoas_em_casa'),
            renda_familiar_mensal=request.form.get('renda_familiar_mensal'),
            escola=request.form.get('escola'),
            uf_prova=request.form.get('uf_prova'),
            possui_celular_em_casa=request.form.get('possui_celular_em_casa'),
            possui_computador_em_casa=request.form.get('possui_computador_em_casa'),
            sexo=request.form.get('sexo'),
            treineiro=request.form.get('treineiro'),
            lingua=request.form.get('lingua'),
            acesso_internet_em_casa=request.form.get('acesso_internet_em_casa')
        )

        input_df = input_data.get_input_data_df()
        print(input_df)
        print('\nBefore prediction.')

        predict_pipeline = PredictPipeline()
        print('\nMid prediction')
        prediction = predict_pipeline.predict(input_df)
        print('\nAfter prediction.')

        return prediction
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)