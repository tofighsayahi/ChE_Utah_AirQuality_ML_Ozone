# -*- coding: utf-8 -*-
"""
Title : Trainer Functions

Description : This the functions that help make it a bit easier to use Keras 
and TensorFlow and divides your dataset for you

Function Dependencies:
keras
numpy
random
TensorFlow
os 
python
    
Created on Tue Jun 16 21:36:12 2019
Revised and Commented 08/15/2019
@author: Timothy Quah
"""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np
import random
from keras.models import load_model
import os


"""
Title:
r2_keras

Description: Used to compute metric R^2 
    
Function Dependencies:
Python 3
keras

Inputs:
y_true- Actual output value dtype = keras object
y_pred- Predicted output value dtype = keras object

Optional Inputs:
None 

Outputs:
r^2 value dtype = keras object

Optional Outputs:
None 
"""



def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

"""
Title:
model_neural_network

Description: Used to initialize neural network
    
Function Dependencies:
Python 3
keras
TensorFlow

Inputs:
layer - number of layers dtype = int
nodes- number of nodes dytpe = int
input_dim_- input dimensions dytpe = int
output_dim_ output dimensions dtype = int

Optional Inputs:
hidden_layer - type of hidden layer activation function dtype = str
activation_layer - type of output activation function dtype = str
DropPercent - drop rate for drop out normalization dtype = float

Outputs:
model - Neural network dtype-tensorflow object

Optional Outputs:
None 
"""



def model_neural_network(layer,nodes,input_dim_,output_dim_,\
                         hidden_layer = 'relu',activation_layer = 'linear',\
                         DropPercent = 0.1):
    #initializes the model
    model = Sequential()
    #add input layers
    model.add(Dense(nodes, input_dim=input_dim_, activation=hidden_layer))
    
    for i in range(0,layer,1):
    #add hidden layers
        model.add(Dropout(DropPercent))
    #add output layer
    model.add(Dense(output_dim_, activation=activation_layer))
    return model




"""
Title:
norm_divider

Description: Randomely splits dataset into training and validation
    
Function Dependencies:
Python 3
numpy 
random 

Inputs:
Data_Array - data array that should be split dtype = numpy array

Optional Inputs:
train_percent - percent training dtype = float
rnum -random seed dtype = int

Outputs:
train_list - the index of training data dtype = list
valid_list - the index of validation data dtype = list

Optional Outputs:
None 
"""



def norm_divider(Data_Array,train_percent=0.7,rnum = 7):
    #initalize random seed
    random.seed(rnum)
    np.random.seed(rnum)
    #get length of data
    data_len = len(Data_Array[:,0])
    train_len = int(np.round(data_len*train_percent))
    #scramble the data array
    scramble_array = np.array(random.sample(range(data_len), data_len))
    #assign part to train and part to valid
    train_list = scramble_array[0:train_len]
    valid_list = scramble_array[train_len:]
    return train_list,valid_list


"""
Title:
divider_XY

Description: Actually divides the dataset returning the training and validation
datasets
    
Function Dependencies:
Python 3
numpy 
 

Inputs:
Data_Array - data array that should be split dtype = numpy array
X_list - x locations of data dtype = list
Y_list  - y locations of data dtype = list
train_list - the index of training data dtype = list
valid_list - the index of validation data dtype = list

Optional Inputs:
None

Outputs:
X-Training data dtype = numpy array
Y- Training data dtype = numpy array
X_valid - validation data dtype = numpy array
Y_valid - validation data dytpe = numpy array


Optional Outputs:
None 
"""

def divider_XY(X_list,Y_list,Data_Array,train_list,valid_list):
    #split X and Y
    X_RH_norm = Data_Array[:,X_list]
    Y_RH_norm = Data_Array[:,Y_list]
    #Split off training
    X =X_RH_norm[train_list]
    Y =Y_RH_norm[train_list]
    #split off validation
    X_valid = X_RH_norm[valid_list]
    Y_valid = Y_RH_norm[valid_list]
    return X,Y,X_valid,Y_valid

"""
Title:
load_evaluate_neural_net

Description: Loads and evaluates a neural network
    
Function Dependencies:
Python 3
keras 
TensorFlow 
os 
r2_keras
numpy 

Inputs:
path - path where the neural network is saved dtype = str
name - name of the neural network dtype = str
X_input - numpy array of X data dtype = numpy array

Optional Inputs:
None

Outputs:
Y_pred - numpy array of prediction dtype = numpy array


Optional Outputs:
None 
"""

def load_evaluate_neural_net(path,name,X_input):
    #get path
    load_model_path = os.path.join(path,name)
    #load model
    model = load_model(load_model_path,custom_objects={'r2_keras':r2_keras})
    #evaluate model
    Y_pred = model.predict(X_input)
    return Y_pred

"""
Title:
mse

Description: Used to compute loss MSE
    
Function Dependencies:
Python 3
numpy 

Inputs:
y_true- Actual output value dtype = numpy array
y_pred- Predicted output value  dtype = numpy array

Optional Inputs:
None 

Outputs:
MSE value  dtype = numpy array

Optional Outputs:
None 
"""



def mse(Y,Ypred):
    n = len(Y)
    return (1/n)*np.mean((Ypred-Y)**2)

"""
Title:
r2

Description: Used to compute metric R^2 
    
Function Dependencies:
Python 3
numpy 

Inputs:
y_true- Actual output value dtype = numpy array
y_pred- Predicted output value  dtype = numpy array

Optional Inputs:
None 

Outputs:
MSE value  dtype = numpy array

Optional Outputs:
None 
"""

def r2(Y, Y_pred):
    SS_res = np.sum((Y-Y_pred)**2)
    SS_tot = np.sum((Y-np.mean(Y))**2)
    return ( 1 - SS_res/(SS_tot + 1e-6) )
