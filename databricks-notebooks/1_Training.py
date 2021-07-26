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
# MAGIC Data source: Random sampled numbers between (1, 50000) with random noise.
# MAGIC 
# MAGIC 
# MAGIC <img src ='https://onebigdatabag.blob.core.windows.net/sparkdemo/tp_ml.jpg?sv=2020-04-08&st=2021-07-18T21%3A40%3A05Z&se=2021-07-30T21%3A40%3A00Z&sr=b&sp=r&sig=3Tu%2Ff2KzNHlBE0O4zYUKceUvR5DIgsFh2RyxcrE0RdQ%3D' alt="Traditional Programming vs Machine Learning" width='1000px'>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src ='https://onebigdatabag.blob.core.windows.net/sparkdemo/KM_achitecture.jpg?sv=2020-04-08&st=2021-07-21T03%3A01%3A59Z&se=2021-07-31T03%3A01%3A00Z&sr=b&sp=r&sig=D3eqep9hW5aReuFzj%2F1AQu1VwxPfXD7EdA%2FYyBi%2B8u4%3D' alt="MLOps Architecture" width='600px'>

# COMMAND ----------

# MAGIC %run "./scripts/init"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Step 1 : Model Training
# MAGIC 
# MAGIC **Algorithms**
# MAGIC - Linear Regression
# MAGIC - 2 hidden layer NN
# MAGIC 
# MAGIC 
# MAGIC <img src ='https://onebigdatabag.blob.core.windows.net/sparkdemo/mlflow_train.jpg?sv=2020-04-08&st=2021-07-19T12%3A48%3A22Z&se=2021-07-30T12%3A48%3A00Z&sr=b&sp=r&sig=eAR570f3wa7efGpu1ohZ3tvlc%2FOFLmLau84Ihrrvosg%3D' width='600px'>

# COMMAND ----------

experiment_name = "/Shared/mlflow-demo" 
mlflow.set_experiment(experiment_name)

# COMMAND ----------

#Prepare data
trainDF, testDF = load_data(5000)
display(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline : Linear Regression

# COMMAND ----------

with mlflow.start_run(run_name="linear-model") as run:

  mlflow.log_param("trainDataSize", trainDF.count())
  
  # training
  featureCols = [col for col in trainDF.columns if col != 'km']
  vecAssembler = VectorAssembler(inputCols=featureCols, outputCol="features")
  lr = LinearRegression(featuresCol="features", labelCol="km")
  stages = [vecAssembler, lr]
  pipeline = Pipeline(stages=stages)
  model = pipeline.fit(trainDF)
  
  # Log model
  mlflow.spark.log_model(model, "linear", input_example=trainDF.limit(5).toPandas()) 
  
  # Evaulation
  regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="km", metricName="rmse")
  rmse = regressionEvaluator.evaluate(model.transform(testDF))
  mlflow.log_metric("rmse", rmse)
  
  # Plot
  evaluation_plot(model, testDF)
  mlflow.log_artifact("eval.png")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Build a TensorFlow FCN model

# COMMAND ----------

# enable MLflow autolog
mlflow.tensorflow.autolog()

# define model
model = Sequential()
model.add(Dense(8, input_shape=(1,)))
model.add(Dense(16))
model.add(Dense(1))
# compile the model
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# prepare data
train_dataset, test_dataset = load_tf_dataset(trainDF, testDF)

# fit the model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=500, verbose=0)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Step 2 : Model Selection
# MAGIC 
# MAGIC In this step, we are selecting the best trained model by **RMSE** recorded in MLflow tracking and register it in **Azure Machine Learning service**
# MAGIC 
# MAGIC 
# MAGIC ### 2.1 Select best model
# MAGIC 
# MAGIC <img src ='https://onebigdatabag.blob.core.windows.net/sparkdemo/mlflow_model_selection.jpg?sv=2020-04-08&st=2021-07-19T12%3A52%3A50Z&se=2021-07-30T12%3A52%3A00Z&sr=b&sp=r&sig=R%2FmvwRDIXGgCIfuB%2FuWsssFqVyCkMcpBe1PSQiLBPtw%3D' width='600px'>

# COMMAND ----------

from mlflow.tracking import MlflowClient

# Create an experiment with a name that is unique and case sensitive.
client = MlflowClient()

experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
run_list = client.search_runs(experiment_id)

end_timestamp = 0
rmse = 9999999
for r in run_list:
  run_info = r.to_dictionary()
  if r.info.status == 'FINISHED' and r.info.end_time > end_timestamp and rmse >= r.data.metrics['rmse']:
    last_run_id = r.info.run_id
    rmse = r.data.metrics['rmse']

model_uri = f"runs:/{last_run_id}/model"
print(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Register Model and build a container image
# MAGIC 
# MAGIC 
# MAGIC <img src ='https://onebigdatabag.blob.core.windows.net/sparkdemo/mlflow_model_registry.jpg?sv=2020-04-08&st=2021-07-19T12%3A53%3A20Z&se=2021-07-30T12%3A53%3A00Z&sr=b&sp=r&sig=%2BdOJAj7PVEaiwh5jDGzuYQb9oUs0c9ApYwbBMooYnGA%3D' width='600px'>

# COMMAND ----------

service_principal_clientid = dbutils.secrets.get(scope = "secret", key ="service-principal-client-id") # Service Principal ID
service_principal_secret = dbutils.secrets.get(scope = "secret", key ="service-principal-secret") # Service Principal Secret
subscription_id = dbutils.secrets.get(scope = "secret", key ="subscription-id") 
tenant_id = dbutils.secrets.get(scope = "secret", key ="tenant-id") 

# Connect to Azure Machine Learning space
azureml_workspace = Workspace(
       subscription_id=subscription_id,
       resource_group='sparkdemorg',
       workspace_name='sparkml',
       auth=service_principal_auth(tenant_id, service_principal_clientid , service_principal_secret))

# Link to Azure Machine Learning 
uri = azureml_workspace.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(uri)


# Build an Azure ML Container Image for an MLflow model
# This step could take 8 - 10 minutes
azure_image, azure_model = mlflow.azureml.build_image(model_uri=model_uri,
                                                      workspace=azureml_workspace,
                                                      image_name='km-predictor-image',
                                                      model_name='km-predictor-model',
                                                      description='Ready to be deployed',
                                                      synchronous=True)
# If your image build failed, you can access build logs at the following URI:
print("Access the following URI for build logs: {}".format(azure_image.image_build_log_uri))


# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3 : Model Deployment
# MAGIC 
# MAGIC In this step, model registry will trigger an automatic deployment to production. A published Azure Machine Learning pipeline will be triggered to create a web service using recent registered model.
# MAGIC 
# MAGIC 
# MAGIC <img src ='https://onebigdatabag.blob.core.windows.net/sparkdemo/mlflow_production.jpg?sv=2020-04-08&st=2021-07-19T12%3A56%3A03Z&se=2021-07-30T12%3A56%3A00Z&sr=b&sp=r&sig=SQaPCf%2BV079Svw6C4h6WH3mj9OF60ssCUxDtX09s%2F3Y%3D' width='600px'>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src ='https://onebigdatabag.blob.core.windows.net/sparkdemo/ReleasePipeline.jpg?sv=2020-04-08&st=2021-07-19T17%3A43%3A36Z&se=2021-07-30T17%3A43%3A00Z&sr=b&sp=r&sig=WCdx%2FNBASXRug1wzxxPXjuiaNQOJFa%2BEtopzDP3DlMA%3D' width='600px'>
# MAGIC 
# MAGIC <img src ='https://onebigdatabag.blob.core.windows.net/sparkdemo/DevOpsPublishedPipeline.jpg?sv=2020-04-08&st=2021-07-19T17%3A41%3A32Z&se=2021-07-30T17%3A41%3A00Z&sr=b&sp=r&sig=sJ%2FVzFGWLEggQ2GtABvMrvK3LxplHp25uagQhlACP4I%3D' width='600px'>
