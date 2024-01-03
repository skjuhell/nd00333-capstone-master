import numpy as np
import argparse
import os
import pandas as pd

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import Adam

import tensorflow as tf

from azureml.core import Run


print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)



parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--number-epochs', type=int, dest='number_epochs', default=10, help='number of epochs for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                    help='# of neurons in the second layer')

args = parser.parse_args()


train=pd.read_csv("train.csv",index_col=0)
test=pd.read_csv("test.csv",index_col=0)
print(train)
target="median_house_value"
x_train=train.drop(target,axis=1)
y_train=train[target]
x_test=test.drop(target,axis=1)
y_test=test[target]

x_train_values=x_train.values
x_test_values=x_test.values
y_train_values=y_train.values
y_test_values=y_test.values


#################### Build a simple MLP model #####################################

run = Run.get_context()

n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_epochs = args.number_epochs
batch_size = args.batch_size
input_shape=(x_train.shape[1],)

run.log('Batch Size', batch_size)
run.log('Epochs', n_epochs)
run.log('First hidden layer', n_h1)
run.log('Second hidden layer', n_h2)

#  MLP model with 2 hidden layers
model = Sequential()
model.add(Dense(n_h1, input_shape=input_shape, activation='relu'))
model.add(Dense(n_h2, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(x_train_values, y_train_values,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    validation_data=(x_test_values, y_test_values),
                    verbose=1)

score = model.evaluate(x_test_values, y_test_values, verbose=0) # returns the loss value & metrics values (here mae) for the model in test mode

print('Test loss:', score[0])
print('Test mae:', score[1])

run.log('Loss', score[0])
run.log('MAE', score[1])

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")
