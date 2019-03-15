from __future__ import division

# from matplotlib.pyplot import step, xlim, ylim, show
# import matplotlib.pyplot as plt
import datetime
import pytz

from datetime import timedelta
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# xbos clients
from xbos import get_client
from xbos.services.hod import HodClient
from xbos.services.mdal import *

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from house import IEC
import IPython
import re
import nn_testing
from nn_testing import IEC_nn

#########################
# New iteration process #
#########################

current_site = "orinda-community-center"

meter_data = (pd.read_csv("power_tstat_data/"+current_site+".csv"))

meter_data["time"] = [re.sub("-08:00","",time) for time in meter_data["Unnamed: 0"]]
meter_data[["time"]] = [pd.to_datetime(time, format = "%Y-%m-%d %H:%M:%S") for  time in meter_data["time"]]

temp_date = meter_data["time"].map(lambda x: 10000*x.year + 100*x.month + x.day)
unique_dates = list(set(temp_date))
meter_data["date"] = temp_date 

number_of_test_days = 3

sampled_idx = sorted(random.sample(range(int(len(unique_dates)/2), len(unique_dates)), number_of_test_days))
sampled_dates = [unique_dates[idx] for idx in sampled_idx]

meter_data_15 = meter_data.resample("15T", on="time").mean()
meter_data = meter_data.set_index("time")
#meter_data = meter_data.tz_localize("utc")
#meter_data = meter_data.tz_convert(pytz.timezone("America/Los_Angeles"))
meter_data["House Consumption"] = meter_data[current_site]
meter_data_15["House Consumption"] = meter_data_15[current_site]

algo_name = "ARM_rnn"
error= []

for i, test_date in enumerate(sampled_dates):
    if test_date < 20180225:
        print("Sampled date is too small")
        continue

    else: 
        print(test_date)

        training_data = meter_data[meter_data["date"]<=test_date]
        test_data = meter_data_15[meter_data_15["date"]== test_date]

#        training_data = training_data.sort_index(ascending = True)

        model = IEC_nn(training_data, prediction_window = test_data.shape[0])
        prediction = model.predict([algo_name])


        error_i = np.sqrt(np.mean((test_data["House Consumption"].values-prediction[algo_name].values)**2))
        error.append(error_i)

        if i == 1: 
            future_window = test_data.shape[0]
            index = np.arange(test_data.shape[0])
            plt.title(algo_name)
            plt.plot(index, prediction[[algo_name]], label="Energy Prediction")
            plt.plot(index, meter_data_15[["House Consumption"]][-future_window:], label="Ground Truth")
            plt.xlabel('Predictive horizon (Minutes)')
            plt.ylabel(r'KWh')
            plt.legend()
            plt.savefig(algo_name+'.png')

print("---------error----------")
print(error)
print(np.mean(error))


