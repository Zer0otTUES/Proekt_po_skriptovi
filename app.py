from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Here you'd get the stock data from somewhere
    # You can use request.get_json() to get any data sent with the POST request
    
    # The LSTM model creation and prediction code would go here
    
    # Return the prediction result
    return str(closing_price)

if __name__ == "__main__":
    app.run(debug=True)
