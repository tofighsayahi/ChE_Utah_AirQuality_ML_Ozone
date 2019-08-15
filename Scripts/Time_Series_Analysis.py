# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:01:39 2019

@author: Tim
"""
import numpy as np
from sklearn.externals import joblib
import pandas as pd
from keras.models import load_model
import os
function_path = "D:/Python_Repository/che_utah_air_quality_group/Functions"
os.chdir(function_path)
from Trainer_Functions import r2_keras,r2,mse
from Analyzer_Functions import export_graphs
import matplotlib.cm as cm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
plt.close('all')
#import neural network
import_nnet_path =\
r"D:\AirQuality _Research\Neural_Network\Reduce_Params_Redo\PM1_PM10_SWD\remove_PM1_SWD_PM10_optimal_nnet_node_mse_150layer_5new.h5"
#import normalization 
import_norm_path =\
r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\PM1_SWD_PM10\large_data_norm.pkl"
#import data
import_data_path =\
r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\PM1_SWD_PM10\All_Data_norm.csv"




#read data in 
df = pd.read_csv(import_data_path)
df['date'] =  pd.to_datetime(df['date'])
date = np.array(df['date'].copy())
Sensor_List =df[['Sensor']].values.T.tolist()[0]
Sensor_name = list(set(Sensor_List))
del df['Sensor']
del df['date']
header = list(df)
#load normalization 
s = joblib.load(import_norm_path)



params_target = 'O3 Value'
index = header.index(params_target)

header_num = len(header)
Full_List = list(np.arange(0,header_num-1+1e-6,1,dtype=int))
Y_Loc =  header.index(params_target)
Y_header_list = []
Y_header_list.append(Y_Loc)
X_header_list = list(set(Full_List)-set(Y_header_list))



data_array = np.array(df)
Y = data_array[:,Y_header_list]
X = data_array[:,X_header_list]
month = np.array(df['Month'])
month_diff = month[1:]-month[0:-1]
sensor_loc = np.where(month_diff<0)[0]
sensor_loc+=1
sensor_list = []
sensor_list.append(0)
sensor_list += list(sensor_loc)

model = load_model(import_nnet_path,custom_objects={'r2_keras':r2_keras})
Y_pred = model.predict(X)
error = np.sqrt((Y_pred-Y)**2)


Stack_Actual = data_array.copy()
Stack_Predict = data_array.copy()
Stack_Predict[:,Y_Loc] = Y_pred.ravel()

Data_Array_Actual_invnorm = s.inverse_transform(Stack_Actual)
Data_Array_Predict_invnorm = s.inverse_transform(Stack_Predict)

Y_pred_ppm = Data_Array_Predict_invnorm[:,Y_Loc]
Y_act_ppm = Data_Array_Actual_invnorm[:,Y_Loc]


error_ppm = np.sqrt((Y_pred_ppm-Y_act_ppm)**2).reshape(-1,1)
max_error_ppm = np.max(error_ppm)
min_error_ppm = np.min(error_ppm)

#
for i in range(0,len(sensor_list)-1,1):
    fig = plt.figure(figsize=(15,6))
    plt.scatter(date[sensor_list[i]:sensor_list[i+1]],\
                error[sensor_list[i]:sensor_list[i+1]],s=1.0,c = error[sensor_list[i]:sensor_list[i+1]],cmap = cm.inferno,vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.title('Time Series of Sensor:'+Sensor_name[i])
    plt.tight_layout()
    name = 'Time_Series'+Sensor_name[i]
    export_graphs(name,fig,filetype='.jpg')


    fig = plt.figure(figsize=(15,6))
    plt.scatter(date[sensor_list[i]:sensor_list[i+1]],\
                error_ppm[sensor_list[i]:sensor_list[i+1]],s=1.0,c = error_ppm[sensor_list[i]:sensor_list[i+1]],cmap = cm.inferno,vmin=min_error_ppm, vmax=max_error_ppm)
    plt.colorbar()
    plt.xlabel('Date')
    plt.ylabel('Error (ppm)')
    plt.title('Real Variable Time Series of Sensor:'+Sensor_name[i])
    plt.tight_layout()
    name = 'Real_Variable_Time_Series'+Sensor_name[i]
    export_graphs(name,fig,filetype='.jpg')
    
    fig = plt.figure(figsize=(15,6))
    plt.scatter(date[sensor_list[i]:sensor_list[i+1]],\
                Y_act_ppm[sensor_list[i]:sensor_list[i+1]],s=1.0,c = 'k',label = 'Data')
    plt.scatter(date[sensor_list[i]:sensor_list[i+1]],\
            Y_pred_ppm[sensor_list[i]:sensor_list[i+1]],s=1.0,c = 'r',label = 'Prediction')
    
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Ozone (ppm)')
    plt.title('Real Variable Time Series of Sensor:'+Sensor_name[i])
    plt.tight_layout()
    name = 'Real_Variable_Time_Series_actual_data_'+Sensor_name[i]
    export_graphs(name,fig,filetype='.jpg')

