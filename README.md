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
In this project, our primary objective is to address a regression problem—a scenario where a model is designed to learn and predict a continuous numerical output in response to input data. We commence by employing AutoML, wherein multiple models undergo training to ascertain their efficacy in fitting the provided training data. The model exhibiting the highest performance score is then selected and preserved as the optimal choice.

Following this, we delve into the construction of a neural network featuring two hidden layers. This neural network is implemented using the Keras framework, and the hyperparameters are fine-tuned using HyperDrive to optimize its predictive capabilities.


## Dataset  <a name="dataset"></a>
In this project we consider the *California housing* data set from [kaggle](https://www.kaggle.com/camnugent/california-housing-prices). The data contains information from the 1990 California census. So although it may not help us with predicting current housing prices, we chose the data set because it requires rudimentary data cleaning, has an easily understandable list of variables and sits at an optimal size between being to toyish and too cumbersome. This enables us to focus on all required configuration to work with Azure ML. 

### Task  <a name="task"></a>
Our objective is to build prediction models that predict the housing prices from the set of given house features.

### Access <a name="access"></a>
We download the *housing.csv* from kaggle localy and upload the csv file to the Azure ML platform.

## General Set Up <a name="setup"></a>
Before we either apply Automated ML or tune hyperparamteres for a keras model, the following steps are required:
For this project, the initial steps involve importing all the necessary dependencies and setting up a Workspace while initializing an experiment. To optimize computational resources, it's crucial to create or attach specific compute resources—employing a VM with a CPU for automated ML and HyperDrive. Data preparation includes loading the CSV file, with the option to either register the dataset in the Dataset section or read it directly into the notebook. Subsequently, one must initialize the AutoMLConfig/HyperDriveConfig object, submit the experiment for processing, and save the best model acquired during the experiment. The final stages revolve around deploying and consuming the best model, whether it stems from the automated ML or HyperDrive run. This comprehensive approach ensures a systematic and efficient workflow throughout the project.


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
![rund_detail1](https://github.com/skjuhell/nd00333-capstone-master/blob/main/screenshots/automl_widget.png)

We collect and save the best model, that is, keras model with the tuned hyperparameters which yield the lowest mean absolute error:
 ``` 
best_run=hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
 ``` 
For a complete code overview, we refer to the jypter notebook *automl.ipynb*.

### Results <a name="automl_result"></a>
The best model from the automated ML run is *Voting Ensemble* displayed below
![rund_detail1](https://github.com/skjuhell/nd00333-capstone-master/blob/main/screenshots/best_model_automl.png)

We can obtain the MAE from the model by calling 
 ```
automl_run_metrics = automl_run.get_metrics()
automl_run_metrics
 ``` 
which returns
```
{'normalized_root_mean_squared_error': 0.09619700249370727,
 'root_mean_squared_log_error': 0.23057708337755914,
 'r2_score': 0.8338871313629255,
 'spearman_correlation': 0.9200313495801759,
 'root_mean_squared_error': 46655.738603453014,
 'explained_variance': 0.8339048748637179,
 'median_absolute_error': 20234.03860918009,
 'normalized_root_mean_squared_log_error': 0.06575588571652256,
 'normalized_mean_absolute_error': 0.06389772701377223,
 'mean_absolute_error': 30990.52539713356,
 'mean_absolute_percentage_error': 17.46132751931706,
 'normalized_median_absolute_error': 0.041719495196267414
}
```

According to the upper output we obtain a  mean absolute error (mae) of 30990.53. 


### Thoughts about improvement <a name="automl_imrpove"></a>
Customized Featurization:

Explore improved featurization by leveraging a FeaturizationConfig object within the AutoMLConfig class's featurization parameter.
This approach allows fine-tuning, enabling the selection of specific encoders for categorical variables, strategies for imputing missing values, and more.

## Hyperparameter Tuning <a name="hyperdrive"></a>
We will compare the above automl run with an ANN with two hidden layers. We tune the following hyperparamters with HyperDrive:
- batch size,
- number of epochs,
- number of units for the two hidden layers.

To initialize a HyperDriveConfig we first need to define the search space
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
- Early termination policy: BanditPolicy ensuring that computational resources are used efficiently by terminating runs that are not likely to lead to substantial improvements in the model's performance.
```
policy =  BanditPolicy(evaluation_interval=2, slack_factor=0.1, slack_amount=None, delay_evaluation=0)
```

- A ScriptRunConfig for setting up configuration for script/notebook runs (here we use the script "keras_train.py"). Since we are using a keras model, we also need to set up an environment.

```
# Evironment set up
# conda_dependencies.yml is stored in the working directory

from azureml.core import Environment

keras_env = Environment.from_conda_specification(name = 'keras-2.3.1', file_path = 'conda_dependencies.yml')


# ScriptrunConfig 
src = ScriptRunConfig(source_directory=project_folder,
                      script='keras_train.py',
                      compute_target=compute_target,
                      environment=keras_env)
 ```     
In configuring an experiment, it's essential to define the primary metric, denoted as MAE (Mean Absolute Error), and specify whether the objective is to maximize or minimize this metric. Additionally, considerations for resource management involve setting the maximum total runs, determining the highest number of runs to be executed concurrently, and acknowledging that concurrent runs are contingent on the available resources in the designated compute target. It is crucial to ensure that the chosen compute target possesses adequate resources to support the desired level of concurrency, aligning the experiment execution with the specified constraints for optimal efficiency.

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
![rundetails_hyperdrive](https://github.com/skjuhell/nd00333-capstone-master/blob/main/screenshots/hyperdrive_widget.png) 

We collect and save the best model, that is, keras model with the tuned hyperparameters which yield the lowest mean absolute error:
 ``` 
best_run=hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
 ``` 
 
### Results <a name="hyperdrive_result"></a>
Here are the results of our hyperdrive run, that is, the tuned hyperparameters and mean absolute error:
 ``` 
{'Epochs': 10,
 'Batch Size': 15,
 'First hidden layer': 4,
 'Second hidden layer': 4,
 'Loss': 17692137654.573643,
 'MAE': 95059.7109375}
  ``` 
These are taken from the *hyperparameter_tuning.ipynb* 
![best_model_hyperdrive](https://github.com/skjuhell/nd00333-capstone-master/blob/main/screenshots/best_model_hyperdrive.png) 


### Thoughts about improvement <a name="hyperdrive_improve"></a>
To enhance the mean absolute error score of our model, we can implement several strategies. First, opting for a more exhaustive Grid Sampling strategy during hyperparameter tuning provides a comprehensive exploration of the parameter space, potentially uncovering optimal configurations. Additionally, we can focus on fine-tuning specific aspects, such as fixing the number of epochs and concentrating on tuning the learning rate for the Keras optimizer, specifically 'adam.' This targeted adjustment allows for a more nuanced optimization of the training process. Moreover, experimenting with the architecture by varying the number of hidden layers presents another avenue for improvement. Tuning the number of hidden layers enables us to identify an architecture that strikes a balance between complexity and performance, ultimately contributing to an enhanced mean absolute error score.

## Deploy Model to ACI <a name="deployment"></a>
With the Keras model exhibiting superior performance after hyperparameter tuning—manifesting as a notably lower mean absolute error we are now ready to proceed with its deployment. The deployment process entails the creation of essential components. Firstly, a scoring script (scoring.py) must be formulated to handle web service calls. This script necessitates two imperative functions: init(), responsible for loading the model into a global object during Docker container initiation, and run(input_data), utilizing the model to predict values based on input data. Secondly, an environment file (myenv.yml) is required to specify the conda packages essential for the Docker image utilized by Azure Machine Learning during deployment. Specifically, TensorFlow and Keras must be outlined in the environment file. These meticulous preparations lay the groundwork for a seamless deployment, ensuring that the refined model is integrated effectively into the Azure Machine Learning environment with the requisite dependencies.

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
![best_model_hyperdrive](https://github.com/skjuhell/nd00333-capstone-master/blob/main/screenshots/endpoints_keras.png) 

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

## Screen Recording  <a name="recording"></a>
[Screencast](https://www.youtube.com/watch?v=YRwcRkIQTh0)

## Standout Suggestions <a name="standout"></a>
(**Optional**)
