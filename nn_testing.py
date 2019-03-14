import pickle
from datetime import timedelta
import datetime as dt
from datetime import datetime
from functools import partial
import os.path

import numpy as np
import keras
import pandas as pd
import scipy.signal
from scipy import spatial
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time 
import pdb
import IPython

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape, TimeDistributed, Dropout, BatchNormalization, Permute, Bidirectional
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

import time


def is_Spring(month):
  if month >3 and month<7:
      return 1
  else: 
      return 0

def is_Summer(month):
    if month> 6 and month < 10:
        return 1
    else: 
        return 0 

def is_Autumn(month): 
    if month > 9:
      return 1 
    else:
      return 0 

def is_Winter(month):
  if month ==12 or month <3:
    return 1 
  else:
    return 0

def timeseries_to_supervised(data, lag=1, steps = 10, dropnan= True, prefix= ""):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(lag, 0, -1):
        cols.append(df.shift(i))
        names += [(prefix + 'var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, steps):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(prefix + 'var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [(prefix + 'var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan: 
        agg.dropna(inplace=True)
    return agg

def is_weekday(day):
    if day in range(0,5):
        return 1
    else:
        return 0


def is_workday(hour):
    if hour in range(8,18):
        return 1
    else:
        return 0

class IEC_nn(object):
    """The Intelligent Energy Component of a house.
    IEC will use several methods to predict the energy consumption of a house
    for a given prediction window using historical data.
    """

    def __init__(self, data, prediction_window=16 * 4):
      """Initializing the IEC.

      Args:
          :param data: Historical Dataset. Last value must be current time
      """
      self.data = data
      self.now = data.index[-1]
      self.prediction_window = prediction_window
      self.algorithms = {
        # "Simple Mean": self.simple_mean,
        # "Usage Zone Finder": self.usage_zone_finder,
        # "ARIMA": self.ARIMAforecast,
        # "Baseline Finder": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=50,
        #                            short_interp_range=25, half_window=50, similarity_interval=5,
        #                            recent_baseline_length=250,
        #                            observation_length_addition=120, short_term_ease_method=easeOutSine,
        #                            long_term_ease_method=easeOutCirc),
          "ARM_rnn": partial(self.ARM_rnn, training_window= 1440*60, k=7, recent_baseline_length=5, num_samples = 2000, nn_file = False, trained_nn_file = "")
        }

    def ARM_rnn(self, training_window= 1440*60, k=7, recent_baseline_length=5, num_samples = 2000, nn_file = False, trained_nn_file = ""):
      training_data = self.data

      training_df = (
          training_data
          .reset_index()
          .rename(columns={'time':'X', 'House Consumption':'y'})
      )

      scalerY = MinMaxScaler(feature_range=(0, 1))
      scalerX = MinMaxScaler(feature_range=(0, 1))
      X= training_df['X']
      X = pd.DataFrame(X)

      s= ["s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12","s13"]
      S = pd.DataFrame(training_df[s]) 
      Y = pd.DataFrame(training_df['y'])

      ## TODO 3/05: add S's so that this is also trained on past actions 

      data = pd.concat([X, S, Y], axis = 1)
      data["is_weekday"] = [is_weekday(x.weekday()) for x in data["X"]]
      data["is_workday"] = [is_workday(x.hour) for x in data["X"]]
      data["is_spring"]= [is_Spring(x.month) for x in data["X"]]
      data["is_summer"]= [is_Summer(x.month) for x in data["X"]]
      data["is_autumn"]= [is_Autumn(x.month) for x in data["X"]]
      data["is_winter"]= [is_Winter(x.month) for x in data["X"]]
      data = data.resample("15T", on="X").mean()
      data.dropna(inplace=True)

      data = data[data["s0"]<.2]
      data = data[data["s1"]<.2]
      data = data[data["s2"]<.2]
      data = data[data["s3"]<.2]
      data = data[data["s4"]<.2]
      data = data[data["s5"]<.2]
      data = data[data["s6"]<.2]
      data = data[data["s7"]<.2]
      data = data[data["s8"]<.2]
      data = data[data["s9"]<.2]
      data = data[data["s10"]<.2]
      data = data[data["s11"]<.2]
      data = data[data["s12"]<.2]
      data = data[data["s13"]<.2]

      data["X"] = data.index
      data["X"] = scalerX.fit_transform(data["X"].reshape(-1,1))
      data["y"] = scalerY.fit_transform(data["y"].reshape(-1,1))

      lag = 48
      steps = self.prediction_window

      ### TODO 3/05: combine S's, X, Y, feature vectors into one dataframe

      Ys_supervised = timeseries_to_supervised(data[["y"]], lag=lag, steps = steps, prefix = "y")
      data.drop("y", axis = 1)

      data_supervised = timeseries_to_supervised(data, lag = lag, steps = 0)
      combined_data = pd.concat([data_supervised, Ys_supervised], axis= 1)
      combined_data.dropna(inplace= True)
      combined_data_values = combined_data.values

      # what is happening here with this X -- why isn't time included? 
      n_test_hours = self.prediction_window
      n_train_hours = len(combined_data_values) - n_test_hours
      n_val_hours = n_train_hours+20

      train = combined_data_values[:n_train_hours, :] 
      val = combined_data_values[n_train_hours:n_val_hours, :]
      test = combined_data_values[n_train_hours:,:]

      print('train',train)

      train_X, train_Y = train[:, :-steps], train[:, -steps:]
      test_X = test[:,:-steps]
      val_X, val_y = val[:, :-steps], val[:, -steps:]

      features=23

      train_X = train_X.reshape((train_X.shape[0], lag, features)) #### TODO: Akaash can you comment? 
      test_X = test_X.reshape((test_X.shape[0], lag, features))
      val_X = val_X.reshape((val_X.shape[0], lag, features))

      batch_size = 72
      epochs=1001
      layers = 5

      if not nn_file:
          trained_nn_file = "arm_orinda_no_bf_rnn_samples_"+ str(batch_size) +"_epochs_"+str(epochs)+"_lag_"+str(lag)+ "_layers_" + str(layers)+ ".json"
          trained_nn_weights_file = "arm_orinda_nn_samples_"+ str(batch_size)+"_epochs_"+ str(epochs)+"_lag_"+str(lag)+ "_layers_" + str(layers)+ ".h5"

      if os.path.isfile(trained_nn_file):
          json_file = open(trained_nn_file, 'r')
          loaded_model_json = json_file.read()
          model = model_from_json(loaded_model_json)
          model.load_weights(trained_nn_weights_file)
          print "Loaded NN model :: ", model
      else:
          print("training!")

          model = Sequential()

          model.add(LSTM(8,  use_bias=True, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
          model.add(Dropout(0.2))
          model.add(LSTM(16,  use_bias=True))
          model.add(Dense(steps, activation='sigmoid'))
          model.compile(loss='mape', optimizer='adam', metrics=['mean_absolute_percentage_error'])

          model.summary()

          # Checkpoint
          filepath="ARC.best.hdf5"
          checkpoint1 = ModelCheckpoint(filepath, monitor='mean_absolute_percentage_error', verbose=1, save_best_only=True, mode='min')
          filepath="ARC_VAL.hdf5"
          checkpoint2 = ModelCheckpoint(filepath, monitor='val_mean_absolute_percentage_error', verbose=1, save_best_only=True, mode='min')

          callbacks_list = [checkpoint1,checkpoint2]


          # Fit the network
          history = model.fit(train_X, train_Y, epochs=epochs, batch_size=72, verbose=1, shuffle=False, callbacks=callbacks_list)

          model_json= model.to_json()
          with open(trained_nn_file, "w") as json_file:
              json_file.write(model_json)
          model.save_weights(trained_nn_weights_file)
          print("saved trained net")

      yhat = model.predict(test_X)



      test_X_New = test_X.reshape((test_X.shape[0], lag*features))

      df = pd.DataFrame()

      for i in range(yhat.shape[1]):
          df['pred_energy_v'+str(i)] = scalerY.inverse_transform(np.concatenate(((yhat[:, i].reshape(yhat.shape[0],1))
                                                                             , test_X_New[:, 1:features]), axis=1))[:,0].tolist()
      return df.values



    def rnn_lstm(self, training_window=1440 * 60, k=7, recent_baseline_length=60, nn_file=False):

        training_data = self.data.tail(training_window)
        # observation_length is ALL of the current day (till now) + 4 hours
        observation_length = mins_in_day(self.now) + 4 * 60

        # IPython.embed()

        training_df = (
            training_data
            .reset_index()
            .rename(columns={'time':'X', 'House Consumption':'y'})
        )

        scaler = MinMaxScaler(feature_range=(0, 1))
        X= training_df['X'].values.reshape(-1,1) # change this if training from energy_prediction

        X = pd.DataFrame(X)
        Y = pd.DataFrame(training_df['y'])

        Xs = (X.fillna(value=0))
        Ys = (Y.fillna(value=0)
#                .pipe(lambda s: (s - s.mean())/s.std()) #### TODO: fix: is this normalization step correct? 
      #          .loc[Xs.index]
                .values.reshape((-1,1))
        )
        #Xs = Xs.values.reshape(-1,1)

        Ys = scaler.fit_transform(Ys)

        lag = 50

        Ys_supervised = pd.DataFrame(timeseries_to_supervised(Ys, lag=lag, steps = 48).values)

        # what is happening here with this X -- why isn't time included? 
        X = Ys_supervised.take(range(0, lag), axis = 1)
        Y = Ys_supervised.take([lag], axis = 1)

        X = X.values.reshape(-1,lag)
#        X["b_predictor"] = training_df["b_predictor"]
        Y = Y.values.reshape(-1,1)
        X = np.reshape(X, (X.shape[0], X.shape[1],1))
   #     Y = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))

        batch_size = 72
        epochs=1
        layers = 5

        if not nn_file:
            trained_nn_file = "orinda_no_bf_rnn_samples_"+ str(batch_size) +"_epochs_"+str(epochs)+"_lag_"+str(lag)+ "_layers_" + str(layers)+ ".json"
            trained_nn_weights_file = "orinda_nn_samples_"+ str(batch_size)+"_epochs_"+ str(epochs)+"_lag_"+str(lag)+ "_layers_" + str(layers)+ ".h5"

        if os.path.isfile(trained_nn_file):
            json_file = open(trained_nn_file, 'r')
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(trained_nn_weights_file)
            print "Loaded NN model :: ", model

        else:
            print("training!")
            # create and fit the LSTM network
            # can try increasing the batch size 

            model = Sequential()
            model.add(LSTM(4, batch_input_shape=(batch_size, lag,1), stateful = True, return_sequences=True))
            model.add(LSTM(4, batch_input_shape=(batch_size, lag,1), stateful = True, return_sequences=True))
            model.add(LSTM(4, batch_input_shape=(batch_size, lag,1), stateful = True))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

            for i in range(2):
                model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle = False)
                model.reset_states()

            model_json= model.to_json()
            with open(trained_nn_file, "w") as json_file:
                json_file.write(model_json)
            model.save_weights(trained_nn_weights_file)
            print("saved trained net")

        prediction_feeder = np.reshape(X[-batch_size:], (X[-batch_size:].shape[0], X[-batch_size:].shape[1],1))
        current_prediction = model.predict(prediction_feeder, batch_size=batch_size)
        # IPython.embed()
        trainPredict = scaler.inverse_transform(current_prediction)[0]#current_prediction.shape[0]-1]

        # is this prediction feeder doing the right thing? (need Anand's help)
        for i in range(self.prediction_window-1):
            prediction_feeder = np.append(prediction_feeder[:,1:lag], current_prediction).reshape(batch_size, lag,1)
            current_prediction = model.predict(prediction_feeder, batch_size=batch_size)
            trainPredict = np.append(trainPredict, scaler.inverse_transform(current_prediction)[0])#current_prediction.shape[0]-1])### maybe the opposite index!!!!

        return trainPredict

    def predict(self, alg_keys):

        index = pd.DatetimeIndex(start=self.now, freq='15T', periods=self.prediction_window)
        result_pred = pd.DataFrame(index=index)
        # IPython.embed()
        for key in alg_keys:
            pred = self.algorithms[key]()
            if (pred.shape[1] if pred.ndim > 1 else 1) > 1: #???
                result_pred[key] = pred[:, 0]
                result_pred[key + ' STD'] = pred[:, 1]
            else:
                result_pred[key] = pred
        return result_pred









