# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:31:37 2019

@author: tofig
"""
import numpy as np
from sklearn.externals import joblib
import pandas as pd
from keras.models import load_model
import os
function_path = "../Functions"
os.chdir(function_path)
from Cleaner_Loader_Functions import indexall
from Trainer_Functions import r2_keras,r2,mse
from Analyzer_Functions import export_graphs,plot_settings
import matplotlib.cm as cm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
plt.close('all')
#import neural network
import_nnet_path =\
r"E:\PhD project\ozone\Saved_Neural_Networks\082119_nnet_node_mse_40layer_1new.h5"
#import normalization 
import_norm_path =\
r"E:\PhD project\ozone\08212019_All_Data_Norm\large_data_norm.pkl"
#import data
import_data_path =\
r"E:\PhD project\ozone\08212019_All_Data_Norm\All_Data_norm.csv"

plot_settings(30,25,25,20,[30,10])

#read data in 
df = pd.read_csv(import_data_path)
df['date'] =  pd.to_datetime(df['date'])
date = np.array(df['date'].copy())
Sensor_List =df[['Sensor']].values.T.tolist()[0]
Sensor_name = list(set(Sensor_List))
#omit datasets for normalization
omit_list = ['date','Sensor']
df_omit = df[omit_list]
for j in range(0,len(omit_list),1):
    del df[omit_list[j]]

header = list(df)
#load normalization 
s = joblib.load(import_norm_path)

params_target = 'O3 Value'
index = header.index(params_target)
#get x and y location
header_num = len(header)
Full_List = list(np.arange(0,header_num-1+1e-6,1,dtype=int))
Y_Loc =  header.index(params_target)
Y_header_list = []
Y_header_list.append(Y_Loc)
X_header_list = list(set(Full_List)-set(Y_header_list))


#divide data
data_array = np.array(df)
Y = data_array[:,Y_header_list]
X = data_array[:,X_header_list]


#load the model
model = load_model(import_nnet_path,custom_objects={'r2_keras':r2_keras})
Y_pred = model.predict(X)

