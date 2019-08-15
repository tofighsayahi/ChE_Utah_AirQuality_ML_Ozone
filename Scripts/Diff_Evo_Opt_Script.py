# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:54:53 2019

@author: Tim
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.optimizers as optimizers
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from functools import partial
from bayes_opt import BayesianOptimization

import_script_path = r"D:\Python_Repository\che_utah_air_quality_group\Functions"
import_data_path = r"D:\AirQuality _Research\Data\Output_Norm_Main_"
#print('Does Data Path? '+str(os.path.exists(import_data_path)))
#print('Does Function Path? '+str(os.path.exists(import_script_path)))

os.chdir(import_script_path)
from Trainer_Functions import r2_keras,\
model_neural_network,load_evaluate_neural_net,norm_divider,divider_XY,mse,r2

plt.close('all')
random.seed(7)
np.random.seed(7)
load_data_list = os.listdir(import_data_path)
#print(load_data_list)


i = 0
full_path = os.path.join(import_data_path,load_data_list[i])
df = pd.read_csv(full_path)
#print(full_path)
    
header =list(df) 

#print(header)
delete_list = ['CO Value','NO2 Value']

for i in range(0,len(delete_list)):
  index = header.index(delete_list[i])
  del df[header[index]]
  del header[index]
#print(header)


header_num = len(header)
Full_List = list(np.arange(0,header_num-1+1e-6,1,dtype=int))
Y_Loc =  header.index('O3 Value')
Y_header_list = []
Y_header_list.append(Y_Loc)
X_header_list = list(set(Full_List)-set(Y_header_list))

data_array = np.array(df)
train_list,valid_list = norm_divider(data_array)
X,Y,X_valid,Y_valid = divider_XY(X_header_list,Y_header_list,data_array,train_list,valid_list)

i = 1
full_path = os.path.join(import_data_path,load_data_list[i])
df_test = pd.read_csv(full_path)

other_header = list(df_test)
for i in range(0,len(delete_list)):
  index = other_header.index(delete_list[i])
  del df_test[other_header[index]]
  del other_header[index]

data_array_test = np.array(df_test)
X_Other = data_array_test[:,X_header_list]
Y_Other = data_array_test[:,Y_header_list]


def all_optimize_function(X,X_valid,X_other,X_header_list,\
                         Y,Y_valid,Y_other,Y_header_list,params):
    layers = int(params[0])
    nodes = int(params[1])
    batchsize = int(params[2])
    epoch = int(params[3])
    input_dim_ = len(X_header_list)
    output_dim_ = len(Y_header_list)
    model = model_neural_network(layers,nodes,input_dim_,output_dim_,\
                                 DropPercent = params[4])
    optimizers.Adam(lr=params[5])
    model.compile(loss='mean_squared_error',optimizer="adam", metrics=[r2_keras])
    model.fit(x=X, y = Y, nb_epoch=epoch, batch_size=batchsize,\
                        verbose=0)
    Y_other_pred = model.predict(X_other)
    Other_MSE = mse(Y_other,Y_other_pred)
    return Other_MSE


optimize_function = partial(all_optimize_function,X,X_valid,X_Other,X_header_list,\
                          Y,Y_valid,Y_Other,Y_header_list)

bounds = [(1,10),(10,300),(10,100),(10,100),(0,0.3),(0.01,0.05)]    



