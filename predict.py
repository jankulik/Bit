import tensorflow as tf
import pandas as pd
import numpy as np

# mean values of “Year Made” and “MachineHours CurrentMeter” from the training dataset
year_mean = 1993.914251
hours_mean = 2672.779697

# load the input data
dataset_original = pd.read_csv("data/input_data.csv",
                               index_col=0,
                               dtype={"datasource": 'string',
                                      "Model Series": 'string',
                                      "Couple System": 'string',
                                      "Grouser Tracks": 'string',
                                      "Hydraulics Flow": 'string'})

# make a copy of input data
dataset = dataset_original.copy()

# drop columns that are not useful as predictors
response_variable = ["Sales Price"]
dataset = dataset.drop(columns=['Sales ID', 'Machine ID', 'Model ID'])

# convert dates to timestamps
dataset["Sales Timestamp"] = pd.to_datetime(dataset["Sales date"]).view('int64') // 10 ** 9
dataset = dataset.drop(columns=["Sales date"])

# specify numerical and categorical variables
numerical_variables = ["Year Made",
                       "MachineHours CurrentMeter",
                       "Sales Timestamp"]
categorical_variables = [variable for variable in dataset.columns
                         if variable not in numerical_variables
                         and variable not in response_variable]

# replace empty or ambiguous entries with “None or Unspecified”
dataset[categorical_variables] = dataset[categorical_variables].where(
    pd.notnull(dataset[categorical_variables]), "None or Unspecified")
dataset = dataset.replace("#NAME?", "None or Unspecified")

# replace 1000s in the “Year Made” column with NaNs
dataset["Year Made"] = dataset["Year Made"].replace(1000, np.nan)

# impute “Year Made” and “MachineHours CurrentMeter” columns with average values from the training dataset
dataset["Year Made"].fillna(value=year_mean, inplace=True)
dataset["MachineHours CurrentMeter"].fillna(value=hours_mean, inplace=True)

# change variable types
dataset[numerical_variables] = dataset[numerical_variables].astype(float)
dataset[categorical_variables] = dataset[categorical_variables].astype('string')

# separate prediction variables
x = dataset.drop(columns=response_variable)

# load model and calculate predictions
model = tf.keras.models.load_model("model/1")
y_predictions = model.predict([x[numerical_variables],
                               x[categorical_variables]],
                              verbose=0).flatten()

# input predictions into the original dataframe and save output
dataset_original["Sales Price"] = y_predictions
dataset_original.to_csv("data/output_data.csv")
