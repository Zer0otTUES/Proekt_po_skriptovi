from flask import Blueprint, render_template, request
import time
import io
import base64
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt

app = Blueprint('app', __name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            company = request.form['company']

            if not company:
                return render_template('predict.html', error='Company name is required.')

            start = '1985-01-01'
            end = '2022-12-31'

            data = yf.download(company, start, end)

            # Ensure there is sufficient data
            if data.empty:
                return render_template('predict.html', error='No data available for the selected company in the selected timeframe.')

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            prediction_days = 60

            x_train = []
            y_train = []

            for x in range(prediction_days, len(scaled_data)):
                x_train.append(scaled_data[x - prediction_days:x, 0])
                y_train.append(scaled_data[x, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            model = Sequential()

            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=25, batch_size=32)

            test_start = '2023-01-01'
            test_end = '2023-06-10'

            test_data = yf.download(company, test_start, test_end)
            actual_prices = test_data['Close'].values

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

            prediction = predicted_prices[-1]

            start_time = time.time()

            plt.figure(figsize=(10, 6))
            plt.plot(actual_prices, color="red", label=f"Actual {company} Price")
            plt.plot(predicted_prices, color="blue", label=f"Predicted {company} Price")
            plt.title(f"{company} Share Price")
            plt.xlabel('Time')
            plt.ylabel(f'{company} Share Price')
            plt.legend()

            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            elapsed_time = time.time() - start_time

            return render_template('predict.html', prediction=prediction, elapsed_time=elapsed_time, plot_url=plot_url)
        
        except Exception as e:
            # Log the error and return an error message
            print(f"An error occurred: {e}")
            return render_template('predict.html', error='An error occurred while processing your request.')

    else:
        return render_template('predict.html')
