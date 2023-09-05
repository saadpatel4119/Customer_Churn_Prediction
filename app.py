from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            CustomerID=request.form.get('CustomerID'),
            Gender=request.form.get('Gender'),
            Location=request.form.get('Location'),
            Age=request.form.get('Age'),
            Subscription_Length_Months=request.form.get('Subscription_Length_Months'),
            Monthly_Bill=float(request.form.get('Monthly_Bill')),
            Total_Usage_GB=float(request.form.get('Total_Usage_GB'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        result1=['Churned' if results[0]==0 else 'Not Churned']
        return render_template('home.html',results=result1[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)