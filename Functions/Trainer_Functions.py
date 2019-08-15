# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:17:49 2019

@author: Tim
"""
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np
import random
from keras.models import load_model
import os


def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def model_neural_network(layer,nodes,input_dim_,output_dim_,\
                         hidden_layer = 'relu',activation_layer = 'linear',\
                         DropPercent = 0.1):
    model = Sequential()
    for i in range(0,layer,1):
        model.add(Dense(nodes, input_dim=input_dim_, activation=hidden_layer))
        model.add(Dropout(DropPercent))
    model.add(Dense(output_dim_, activation=activation_layer))
    return model

def norm_divider(Data_Array,train_percent=0.7,Min = 0,Max = 1,rnum = 7):
    random.seed(rnum)
    np.random.seed(rnum)
    data_len = len(Data_Array[:,0])
    train_len = int(np.round(data_len*train_percent))
    scramble_array = np.array(random.sample(range(data_len), data_len))
    train_list = scramble_array[0:train_len]
    valid_list = scramble_array[train_len:]
    return train_list,valid_list

def divider_XY(X_list,Y_list,Data_Array,train_list,valid_list):
    X_RH_norm = Data_Array[:,X_list]
    Y_RH_norm = Data_Array[:,Y_list]
    X =X_RH_norm[train_list]
    Y =Y_RH_norm[train_list]
    X_valid = X_RH_norm[valid_list]
    Y_valid = Y_RH_norm[valid_list]
    return X,Y,X_valid,Y_valid

def load_evaluate_neural_net(path,name,X_input):
    load_model_path = os.path.join(path,name)
    model = load_model(load_model_path,custom_objects={'r2_keras':r2_keras})
    Y_pred = model.predict(X_input)
    return Y_pred

def mse(Y,Ypred):
    return np.mean((Ypred-Y)**2)

def r2(Y, Y_pred):
    SS_res = np.sum((Y-Y_pred)**2)
    SS_tot = np.sum((Y-np.mean(Y))**2)
    return ( 1 - SS_res/(SS_tot + 1e-6) )
