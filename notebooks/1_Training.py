# Databricks notebook source
# MAGIC %md
# MAGIC # Problem Statement - KM Predictor
# MAGIC 
# MAGIC ### In this problem, we create a machine learning model to predict KM based on miles. 
# MAGIC 
# MAGIC **Input:** Mile (float) 
# MAGIC 
# MAGIC **Output:** KM (float)
# MAGIC 
# MAGIC 
# MAGIC Data source: Random sampled numbers in miles
# MAGIC 
# MAGIC 
# MAGIC <img src ='https://onebigdatabag.blob.core.windows.net/sparkdemo/tp_ml.jpg?sv=2020-04-08&st=2021-07-18T21%3A40%3A05Z&se=2021-07-30T21%3A40%3A00Z&sr=b&sp=r&sig=3Tu%2Ff2KzNHlBE0O4zYUKceUvR5DIgsFh2RyxcrE0RdQ%3D' alt="Traditional Programming vs Machine Learning">

# COMMAND ----------

import warnings
import tensorflow as tf
import keras
import pandas as pd
import numpy as np

# COMMAND ----------

warnings.filterwarnings("ignore", category=DeprecationWarning)

# COMMAND ----------

# MAGIC %md [Source](https://androidkt.com/linear-regression-model-in-keras/)
# MAGIC Modified and extended for this tutorial
# MAGIC 
# MAGIC Problem: Build a simple Linear NN Model that predicts <em>Kilometers</em> from training data with <em>Miles</em> 

# COMMAND ----------

# MAGIC %md Generate our X, y, and predict data

# COMMAND ----------

# miles to km
def miles_to_km(m):
  return m * 1.609344

# generate data sets training, test, prediction
def gen_kmmiles_data(start, stop, length):
  X_miles = [[random.uniform(start, stop)] for i in range(length)]
  y_kms = np.array(np.array([miles_to_km(m) for m in X_miles]))
  predict_data = []
  [predict_data.append(t) for t in range(1,500,5)]
  return (X_miles, y_kms, predict_data)

# COMMAND ----------

# MAGIC %md #### Define Inference Functions

# COMMAND ----------

# function to predict via model and return dataframe of predictions
def predict_keras_model(model, data):
  df = pd.DataFrame(np.array(data))
  predictions = model.predict(df)
  return pd.DataFrame(predictions)



# COMMAND ----------

# MAGIC %md #### Build a Keras Dense NN model

# COMMAND ----------

# Define the model
def baseline_model():
   model = keras.Sequential([
      keras.layers.Dense(64, activation='relu', input_shape=[1]),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(1)
   ])

   optimizer = keras.optimizers.RMSprop(0.001)

   # Compile the model
   model.compile(loss='mean_squared_error',
                 optimizer=optimizer,
                 metrics=['mean_absolute_error', 'mean_squared_error'])
   return model

# COMMAND ----------

def do_train(params, X, y, run_name="Keras Linear Regression MilesToKm"):

    model = baseline_model()
    model.fit(X, y, batch_size=params['batch_size'], epochs=params['epochs'])
    return model


# COMMAND ----------

# MAGIC %md #### Train the Keras model

# COMMAND ----------

# MAGIC %md Generate X, y, and predict_data

# COMMAND ----------

(X, y, predict_data) = gen_kmmiles_data(.5, 10512, 2)

# COMMAND ----------

params = {'batch_size': 10,'epochs': 100}
t = do_train(params, X, y)
print(f"Finished running")

# COMMAND ----------

# do predictions via model
predictions = predict_keras_model(t, predict_data)

# COMMAND ----------

# fix up dataframe by adding prediction set and calculated milet to km
# rename column to prediction_kms
predictions.columns = ['prediction_kms']
# add column and data for original data (miles)
predictions['miles'] = pd.Series(predict_data)
# add calculation for miles to km
predictions['milesTokm_calculated'] = predictions['miles'] * 1.609344

# COMMAND ----------

predictions.head()

# COMMAND ----------

display(predictions)

# COMMAND ----------


