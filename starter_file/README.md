# Train and Deploy Models with Azure ML

## Table of Contents
- [Problem Statement](##problem)
- [Data Set](##dataset)
    - [Task](###task)
    - [Access](###access)
- [Set Up and Installation](##setup)
- [Automated ML](##automl)
    - [Result](##automl_result)
    - [Thoughts about Improvement](##automl_improve)
- [Hyperparameter Tuning](##hyperdrive)
    - [Result](##hyperdrive_result)
    - [Thoughts about Improvement](##hyperdrive_improve)
 - [Model Deployment](##deployment)
 - [Recreen Recording](##recording) 
 - [Standout Suggestions](##standout)

## Problem Statement <a name="problem"></a>
In this project we will consider a regression problem, i.e. a process where a model learns to predict a continuous value output for a given input data. We first apply AutoML where multiple models are trained to fit the training data. We then choose and save the best model, that is, the model with the best score. Secondly, we build a simple neural network consisting of two hidden layers. In particular, a keras model where we tune hyperparameters using HyperDrive.  


## Dataset  <a name="dataset"></a>
In this project we consider the *California housing* data set from [kaggle](https://www.kaggle.com/camnugent/california-housing-prices). The data contains information from the 1990 California census. So although it may not help us with predicting current housing prices, we chose the data set because it requires rudimentary data cleaning, has an easily understandable list of variables and sits at an optimal size between being to toyish and too cumbersome. This enables us to focus on all required configuration to work with Azure ML. 

### Task  <a name="task"></a>
Our objective is to build prediction models that predict the housing prices from the set of given house features.

### Access <a name="access"></a>
We download the *housing.csv* from kaggle localy and upload the csv file to the Azure ML platform.

## General Set Up <a name="setup"></a>
Before we either apply Automated ML or tune hyperparamteres for a keras model, the following steps are required:
- Import all needed dependencies
- Set up a Workspace and initialize an experiment
- Create or attach a compute resource (VM with cpu for automated ML and VM with gpu for hyperdrive)
- Load csv file and either register data set in the *Dataset* section or read directly to the notebook
- Initialize AutoMLConfig / HyperDriveConfig object
- Submit experiment 
- Save best model
- Deploy and consume best model (either for the automated ML or hyperdrive run)


## Automated ML <a name="automl"></a>
To configure the Automated ML run we need to specify what kind of a task we are dealing with, the primary metric, train and validation data sets (which are in *TabularDataset*  form) and the target column name. Featurization is set to "auto", meaning that the featurization step should be done automatically. To avoid overfitting we enable early stopping. 
```
automl_setting={
    "featurization": "auto",
    "experiment_timeout_minutes": 30,
    "enable_early_stopping": True,
    "verbosity": logging.INFO,
    "compute_target": compute_target
}

task="regression" 
automl_config = AutoMLConfig( 
    task=task, 
    primary_metric='normalized_root_mean_squared_error', 
    training_data=train, 
    validation_data = test, 
    label_column_name='median_house_value', 
    **automl_setting
)
```

Next we submit the hyperdrive run to the experiment (i.e. launch an experiment) and show run details with the RunDeatails widget:
 ``` 
automl_run = experiment.submit(automl_config, show_output=True)
RunDetails(automl_run).show()
 ``` 
Screenshots of the RunDetails widget:
![rund_detail1](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/automated_ml/screenshots/automl_run.png)
![run_details2](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/automated_ml/screenshots/autml_run2.png)

We collect and save the best model, that is, keras model with the tuned hyperparameters which yield the lowest mean absolute error:
 ``` 
best_run=hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
 ``` 
For a complete code overview, we refer to the jypter notebook *automl.ipynb*.

### Results <a name="automl_result"></a>
The best model from the automated ML run is *LightGBM* with mean absolute error (mae) of 32.376,38. 

The following screenshots shows the best run ID and mae:
![best_model](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/automated_ml/screenshots/best_model.png)

### Thoughts about improvement <a name="automl_imrpove"></a>
Two thoughts of how we we can perhaps improve the model:
- Use customized featurization by passing a *FeaturizationConfig* object to the featurization parameter of the AutoMLConfig class. This, for example, enables us to choose a particular encoder for the categorical variables, a strategy to impute missing values, ect..
- *LightGBM* is a fast, distributed, high-performance gradient-boosting framework based on decision tree algorithms. We can experiment with different configurations of the model, that is, tune some of the hyperparameters such as *num_leaves*, *learning_rate*, *feature_fraction*,... .

## Hyperparameter Tuning <a name="hyperdrive"></a>
We will compare the above automl run with a deep neural network, in particular, a *keras Sequential model* with two hidden layers. We tune the following hyperparamters with HyperDrive:
- batch size,
- number of epochs,
- number of units for the two hidden layers.

To initialize a HyperDriveConfog class we need to specify the following:
- Hyperparameter space: RandomParameterSampling defines a random sampling over the hyperparameter search spaces. The advantages here are that it is not so exhaustive and the lack of bias. It is a good first choice.
```
ps = RandomParameterSampling(
    {
        '--batch-size': choice(25, 50, 100),
        '--number-epochs': choice(5,10,15),
        '--first-layer-neurons': choice(range(2,12,2)),
        '--second-layer-neurons': choice(range(2,12,2))
    }
)
```
- Early termination policy: BanditPolicy defines an early termination policy based on slack criteria and a frequency interval for evaluation. Any run that does ot fall within the specified slack factor (or slack amount) of the evaluation metric with respect to the best performing run will be terminated. Since our script reports metrics periodically during the execution, it makes sense to include early termination policy. Moreover, doing so avoids overfitting the training data. For more aggressive savings, we chose the Bandit Policy with a small allowable slack.
```
policy =  BanditPolicy(evaluation_interval=2, slack_factor=0.1, slack_amount=None, delay_evaluation=0)
```

- A ScriptRunConfig for setting up configuration for script/notebook runs (here we use the script "keras_train.py"). Since we are using a keras model, we also need to set up an environment.

```
# Evironment set up
# conda_dependencies.yml is stored in the working directory

from azureml.core import Environment

keras_env = Environment.from_conda_specification(name = 'keras-2.3.1', file_path = 'conda_dependencies.yml')

# Specify a GPU base image
keras_env.docker.enabled = True
keras_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.0-cudnn7-ubuntu18.04'

# ScriptrunConfig 
src = ScriptRunConfig(source_directory=project_folder,
                      script='keras_train.py',
                      compute_target=compute_target,
                      environment=keras_env)
 ```     
- Primary metric name and goal: The name of the primary metric reported by the experiment runs (mae) and if we wish to maximize or minimize the primary metric (minimze).
- Max total runs and max concurrent runs : The maximum total number of runs to create and the maximum number of runs to execute concurrently. Note: the number of concurrent runs is gated on the resources available in the specified compute target. Hence ,we need to ensure that the compute target has the available resources for the desired concurrency.

The HyperDriveConfig obejct:
 ``` 
hyperdrive_config = HyperDriveConfig(
    hyperparameter_sampling = ps, 
    primary_metric_name ='MAE', 
    primary_metric_goal = PrimaryMetricGoal.MINIMIZE, 
    max_total_runs = 8, 
    max_concurrent_runs=4, 
    policy=policy, 
    run_config=src
)
 ``` 
Next we submit the hyperdrive run to the experiment (i.e. launch an experiment) and show run details with the RunDeatails widget:
 ``` 
hyperdrive_run = experiment.submit(hyperdrive_config, show_output=True)
RunDetails(hyperdrive_run).show()
 ``` 
Screenshot of the RunDetails widget:
![rundetails_hyperdrive](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/hyperdrive_keras_model/screenshots/hyperdrive_run_completed.png) 

We collect and save the best model, that is, keras model with the tuned hyperparameters which yield the lowest mean absolute error:
 ``` 
best_run=hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
 ``` 
 
### Results <a name="hyperdrive_result"></a>
Here are the results of our hyperdrive run, that is, the tuned hyperparameters and mean absolute error:
 ``` 
{'Batch Size': 25,
 'Epochs': 10,
 'First hidden layer': 4,
 'Second hidden layer': 4,
 'Loss': 50654265725.023254,
 'MAE': 193928.734375}
  ``` 
Here is the screenshot of the best model:
![best_model_keras](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/hyperdrive_keras_model/screenshots/best_model_hyperdirve.png)

### Thoughts about improvement <a name="hyperdrive_improve"></a>
We can perhaps improve the mean absolute error score by:
- choosing the more exhaustive *Grid Sampling strategy*,
- keeping, for example, the number of epochs fixed and tune the hyperparameter *learning rate* for the keras optimizer *adam*,
- choosing a different number of hidden layers, i.e. tune the number of hidden layers as well.

## Deploy Model to ACI <a name="deployment"></a>
The keras model with the tuned hyperparameters archieved a better score, that is, a lower mean absolute error, thus we will deploy that model. To do so, we need to:
- Create a *scoring script* that will be invoked by the web service call (see *scoring.py*). Note that the scoring script must have two required functions, *init()* and *run(input_data)*.
    - An init() function: typically loads the model into a global object. This function is executed only once when the Docker container is started.
    - An run(input_data) function: the model is used to predict a value based on the input data. The input and output to run typically use JSON as serialization and de-serialization format (it is not limited to that).
- Create an *environment file* so that Azure Machine Learning can install the necessary packages in the Docker image which are required by your scoring script. In this case, we need to specify conda packages tensorflow and keras (see *myenv.yml*).

- Create the *inference configuration* and *deployment configuration* and deploy to ACI. 
```
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.model import Model
from azureml.core.environment import Environment


myenv = Environment.from_conda_specification(name="myenv", file_path="myenv.yml")
inference_config = InferenceConfig(entry_script="scoring.py", environment=myenv)

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1,
                                               auth_enabled=True, # this flag generates API keys to secure access
                                               memory_gb=1,
                                               tags={'name': 'housing', 'framework': 'Keras'},
                                               description='Keras MLP on california housing')

service = Model.deploy(workspace=ws, 
                           name='keras-housing-svc', 
                           models=[model], 
                           inference_config=inference_config, 
                           deployment_config=aciconfig)

service.wait_for_deployment(True)
print(service.state)
```
After a successfull deployment we can access the scoring uri with:
```
print(service.scoring_uri)
```
![model_deployed](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/hyperdrive_keras_model/screenshots/keras_model_deployed_healthy.png)

We can test the deployed model. We pick the first 5 samples from the test set, and send it to the web service hosted in ACI. We note here that we are using the run API in the SDK to invoke the service (we can also make raw HTTP calls using any HTTP tool such as curl).
```
import json

test_s=x_test[:5].tolist()
test_samples = json.dumps({"data": test_s})
test_samples = bytes(test_samples, encoding='utf8')

# predict using the deployed model
result = service.run(input_data=test_samples)

from sklearn.metrics import mean_absolute_error
mae_test = mean_absolute_error(y_test[:5], np.array(result))
print(round(mae_test, 3))
```
We can now send construct raw HTTP request and send to the service. Todo so, we need to add a key to the HTTP header:
```
import requests

key1, Key2 = service.get_keys()

# send a random row from the test set to score
random_index = np.random.randint(0, len(x_test)-1)
input_data = "{\"data\": [" + str(list(x_test[random_index])) + "]}"

headers = {'Content-Type':'application/json', 'Authorization': 'Bearer ' + key1}

resp = requests.post(service.scoring_uri, input_data, headers=headers)

print("POST to url", service.scoring_uri)
#print("input data:", input_data)
print("label:", y_test[random_index])
print("prediction:", resp.text)
```

We can have a look at the workspace after the web service was deployed: 
```

model = ws.models['keras-housing']
print("Model: {}, ID: {}".format('keras-housing', model.id))
    
webservice = ws.webservices['keras-housing-svc']
print("Webservice: {}, scoring URI: {}".format('keras-housing-svc', webservice.scoring_uri))
```
```
Model: keras-housing, ID: keras-housing:2
Webservice: keras-housing-svc, scoring URI: http://d16dbe64-9b27-48b9-9cf8-3bc605fec3c7.southcentralus.azurecontainer.io/score
```
At the end we delete the ACI deployment as well as the compute cluster:
![clean_up](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/hyperdrive_keras_model/screenshots/clean_up.png)

## Screen Recording  <a name="recording"></a>
[Screencast](https://youtu.be/05gfBcdG8OQ)

## Standout Suggestions <a name="standout"></a>
(**Optional**)
