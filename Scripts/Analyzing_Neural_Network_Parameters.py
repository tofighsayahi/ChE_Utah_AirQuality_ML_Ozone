# -*- coding: utf-8 -*-
"""
Title : Analyzing Neural Network Parameter

Description : Analyze each parameter's impact on the overall output
Created on Tue Jul  9 22:40:39 2019

@author: Tim
"""
#import packages
import os 
function_path = "../Functions"
os.chdir(function_path)
from Trainer_Functions import r2_keras
from Analyzer_Functions import export_graphs
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

#Setup system
plt.close('all')
random.seed(7)
np.random.seed(7)
#set path to data
full_path =\
r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params_Upload_GoogleDrive\SWD\train_validate_data.csv"

#read data and parse the data
df = pd.read_csv(full_path)    
header =list(df) 
header_num = len(header)
#get X and Y values
Full_List = list(np.arange(0,header_num-1+1e-6,1,dtype=int))
Y_Loc =  header.index('O3 Value')
Y_header_list = []
Y_header_list.append(Y_Loc)
X_header_list = list(set(Full_List)-set(Y_header_list))

#load neural network
load_model_path =\
r"D:\AirQuality _Research\Neural_Network\Reduce_Params_Redo\Remove SWD_Limit_Nodes\remove_swd_nnet_node_mse_16layer_2new.h5"
model = load_model(load_model_path,custom_objects={'r2_keras':r2_keras})
model.summary()



#itterate over each parameter
input_dim_ = len(X_header_list)
rows = 500
range_save = np.zeros(input_dim_)
header_plot1 = []
header_plot2 = []
for i in range(0,input_dim_,1):
    #give average value of 0.5
    X_valid = np.ones([rows,input_dim_])*0.5
    #change one parameter
    X_valid[:,i] = np.linspace(0,1,rows)
    Y_pred = model.predict(X_valid)
    #zero so that it its easier to read
    Y_zero = Y_pred-np.min(Y_pred)
    #plot figure
    fige = plt.figure()
    plt.title(header[X_header_list[i]])
    plt.scatter( X_valid[:,i],Y_zero,c = 'r',s=10)
#    X_Data = df[header[X_header_list[i]]]
#    Y_Data = df['O3 Value']
#    plt.scatter( X_Data,Y_Data,c = 'b',s=1)

    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
#    plt.xlim(-0.1,1.1)
#    plt.ylim(-0.1,1.1)
    plt.tight_layout()
    plot_name = header[X_header_list[i]]+'_'
    #export figure
    export_graphs(plot_name,fig=fige,filetype = '.jpg')
    range_save[i] = np.max(Y_pred)-np.min(Y_pred)
