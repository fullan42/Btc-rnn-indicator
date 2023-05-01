import math
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sklearn.metrics
import keras.optimizers


class BTCValuePredictor:

    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.dataset = pd.read_csv(csv_file_path)
        self.dataset = self.dataset.iloc[:, 4:5]
        self.sc = MinMaxScaler()
        self.days_num = 30
        self.checkpoint_path = 'model_checkpoint.h5'
        self.tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

    def prepare_data(self):
        dataset_scaled = self.sc.fit_transform(self.dataset)
        self.dataset_scaled = pd.DataFrame(dataset_scaled, columns=['Close'])
        reshaped_data = np.zeros((len(self.dataset_scaled) - self.days_num - 1, self.days_num, 1))

        for i in range(0, len(self.dataset_scaled) - self.days_num - 1):
            reshaped_data[i] = self.dataset_scaled.iloc[i:i + self.days_num].values.reshape((self.days_num, 1))

        num = int(len(reshaped_data) * 0.8)
        train_set = reshaped_data[:num]
        test_set = reshaped_data[num:]

        self.xtrain = train_set[0:len(train_set)]
        self.ytrain = self.dataset_scaled[self.days_num:len(train_set) + self.days_num]
        self.xtest = test_set[0:len(test_set)]
        self.ytest = self.dataset_scaled[self.days_num + len(self.xtrain):len(self.dataset_scaled) -1]

        self.ytrain = self.ytrain.to_numpy()
        self.ytest = self.ytest.to_numpy()

    def create_model(self, learning_rate=0.001):
        self.model = Sequential()
        self.model.add(LSTM(units=16, activation='sigmoid', input_shape=(self.days_num, 1), return_sequences=False))
        self.model.add(Dense(units=1))
        opt = keras.optimizers.Adam(learning_rate = learning_rate)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, batch_size=90, epochs=1000):
        tf_callbacks = tf.keras.callbacks.TensorBoard(log_dir="logs/fit", histogram_freq=1)
        checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
        self.history = self.model.fit(self.xtrain, self.ytrain, batch_size=batch_size, epochs=epochs,
                                      validation_split=0.2, callbacks=[checkpoint, tf_callbacks])

    def predict(self):
        predicted_close_price = self.model.predict(self.xtest)
        predicted_close_price = self.sc.inverse_transform(predicted_close_price)
        real_close_price = self.sc.inverse_transform(self.ytest)
        self.predicted_close_price = predicted_close_price
        self.real_close_price = real_close_price
        mse = mean_squared_error(self.real_close_price[1,:], self.predicted_close_price[1,:])
        rmse = math.sqrt(mse)
        predicted_direction = np.diff(self.predicted_close_price)
        actual_direction = np.diff(self.real_close_price)
        mda = np.mean(predicted_direction == actual_direction)
        print("rmse: {}".format(rmse) + " , mda: {}".format(mda))

    def plot_predictions(self):
        plt.plot(self.real_close_price, color='red', label='Real BTC Value')
        plt.plot(self.predicted_close_price, color='blue', label='Predicted BTC Value')
        plt.title('BTC Value Prediction')
        plt.xlabel('Index Num')
        plt.ylabel('BTC Value')
        plt.legend()
        plt.show()

    def plot_loss(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Loss per epoch')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss' , 'val_loss'], loc='upper left')
        plt.show()
