
import base64
import datetime as dt
import io
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request ,Blueprint
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from .auth import login_required

app = Blueprint('app', __name__)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        company = request.form['company']
        start = request.form['start']
        end = request.form['end']
        test_start = request.form['test_start']
        test_end = request.form['test_end']
        future_days = int(request.form['future_days'])

        plot_url, predictions = predict_stock_price(company, start, end, test_start, test_end, future_days=future_days)
        latest_price = get_latest_stock_price(company)
        show_results = True

        return render_template('results.html', show_results=show_results, plot_url=plot_url, predictions=predictions, latest_price=latest_price)
    else:
        return render_template('predict.html')

def predict_stock_price(company, start, end, test_start, test_end, prediction_days=60, future_days=1):
    # Load data
    data = yf.download(company, start=start, end=end)

    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    train_features, train_targets = prepare_training_data(scaled_data, prediction_days)

    # Build the model
    model = build_lstm_model(train_features.shape[1])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_features, train_targets, epochs=25, batch_size=32)

    # Test the model accuracy on existing data
    test_data = yf.download(company, start=test_start, end=test_end)
    actual_prices = test_data['Close'].values

    plot_url, predicted_prices = plot_test_predictions(model, scaler, company, data, test_data, prediction_days)

    # Predict next day
    predictions = predict_future_days(model, scaler, data, test_data, prediction_days, future_days)

    return plot_url, predictions
def prepare_training_data(scaled_data, prediction_days):
    train_features = []
    train_targets = []

    for x in range(prediction_days, len(scaled_data)):
        train_features.append(scaled_data[x - prediction_days:x, 0])
        train_targets.append(scaled_data[x, 0])

    train_features, train_targets = np.array(train_features), np.array(train_targets)
    train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1], 1))

    return train_features, train_targets

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    return model

def plot_test_predictions(model, scaler, company, data, test_data, prediction_days):
    actual_prices = test_data['Close'].values  # Define actual_prices variable here
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plt.clf()
    plt.plot(test_data.index, actual_prices, color="green", label=f"Actual {company} Price")
    plt.plot(test_data.index, predicted_prices, color="red", label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url, predicted_prices

def predict_future_days(model, scaler, data, test_data, prediction_days, future_days):
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    future_predictions = []

    for i in range(future_days):
        model_inputs = total_dataset[len(total_dataset) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)

        x_test = model_inputs[-prediction_days:].reshape(1, -1, 1)

        predicted_price = model.predict(x_test)
        predicted_price = scaler.inverse_transform(predicted_price)
        next_day_prediction = predicted_price[0][0]

        future_predictions.append(next_day_prediction)

        total_dataset = total_dataset.append(pd.Series([next_day_prediction]), ignore_index=True)

    return future_predictions

def get_latest_stock_price(company):
    latest_data = yf.download(company, period='1d')
    latest_price = latest_data['Close'].iloc[-1]
    return latest_price