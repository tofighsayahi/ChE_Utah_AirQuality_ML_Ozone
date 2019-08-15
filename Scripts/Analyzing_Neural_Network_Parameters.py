# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 22:40:39 2019

@author: Tim
"""

import os 
function_path = "D:/Python_Repository/che_utah_air_quality_group/Functions"
os.chdir(function_path)
from Trainer_Functions import r2_keras
from Analyzer_Functions import export_graphs
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
#print('Does Data Path? '+str(os.path.exists(import_data_path)))
#print('Does Function Path? '+str(os.path.exists(import_script_path)))

plt.close('all')
random.seed(7)
np.random.seed(7)
full_path =\
r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params_Upload_GoogleDrive\SWD\train_validate_data.csv"

df = pd.read_csv(full_path)    
header =list(df) 

header_num = len(header)
Full_List = list(np.arange(0,header_num-1+1e-6,1,dtype=int))
Y_Loc =  header.index('O3 Value')
Y_header_list = []
Y_header_list.append(Y_Loc)
X_header_list = list(set(Full_List)-set(Y_header_list))


load_model_path =\
r"D:\AirQuality _Research\Neural_Network\Reduce_Params_Redo\Remove SWD_Limit_Nodes\remove_swd_nnet_node_mse_16layer_2new.h5"
model = load_model(load_model_path,custom_objects={'r2_keras':r2_keras})
model.summary()

input_dim_ = len(X_header_list)
rows = 500
range_save = np.zeros(input_dim_)
header_plot1 = []
header_plot2 = []
for i in range(0,input_dim_,1):
    X_valid = np.ones([rows,input_dim_])*0.5
    X_valid[:,i] = np.linspace(0,1,rows)
    Y_pred = model.predict(X_valid)
    Y_zero = Y_pred-np.min(Y_pred)
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
    export_graphs(plot_name,fig=fige,filetype = '.jpg')
    range_save[i] = np.max(Y_pred)-np.min(Y_pred)
