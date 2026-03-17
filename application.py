from flask import Flask, render_template, request, jsonify
from src.components.data_ingestion import DataIngestionConfig
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')    


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')
            # Get form

    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            writing_score=float(request.form.get('writing_score')),
            reading_score=float(request.form.get('reading_score')),
            
        )
        data_df = data.get_data_as_dataframe()
        print(data_df)  

        predict_pipeline = PredictPipeline()
        results  = predict_pipeline.predict(data_df)
        return render_template('home.html', results=results[0])    

if __name__ == "__main__":
    app.run(host='0.0.0.0')
