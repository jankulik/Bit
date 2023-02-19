import tensorflow as tf
import pandas as pd
import numpy as np

year_mean = 1993.914251
hours_mean = 2672.779697

dataset_original = pd.read_csv("test_data.csv",
                               index_col=0,
                               dtype={"datasource": 'string',
                                      "Model Series": 'string',
                                      "Couple System": 'string',
                                      "Grouser Tracks": 'string',
                                      "Hydraulics Flow": 'string'})

dataset = dataset_original.copy()

response_variable = ["Sales Price"]
dataset = dataset.drop(columns=['Sales ID', 'Machine ID', 'Model ID'])

dataset["Sales Timestamp"] = pd.to_datetime(dataset["Sales date"]).view('int64') // 10 ** 9
dataset = dataset.drop(columns=["Sales date"])

numerical_variables = ["Year Made",
                       "MachineHours CurrentMeter",
                       "Sales Timestamp"]

categorical_variables = [variable for variable in dataset.columns
                         if variable not in numerical_variables
                         and variable not in response_variable]

dataset[categorical_variables] = dataset[categorical_variables].where(
    pd.notnull(dataset[categorical_variables]), "None or Unspecified")

dataset = dataset.replace("#NAME?", "None or Unspecified")

dataset["Year Made"] = dataset["Year Made"].replace(1000, np.nan)

dataset["Year Made"].fillna(value=year_mean, inplace=True)
dataset["MachineHours CurrentMeter"].fillna(value=hours_mean, inplace=True)

dataset[numerical_variables] = dataset[numerical_variables].astype(float)
dataset[categorical_variables] = dataset[categorical_variables].astype('string')

x = dataset.drop(columns=response_variable)

print(dataset.head())

model = tf.keras.models.load_model("model")
y_predictions = model.predict([x[numerical_variables],
                               x[categorical_variables]],
                              verbose=0).flatten()

dataset_original["Sales Price"] = y_predictions
dataset_original.to_csv("test_data_prediction.csv")
