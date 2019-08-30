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
del df['Sensor']
del df['date']
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
month = np.array(df['Month'])
month_diff = month[1:]-month[0:-1]


#load the model
model = load_model(import_nnet_path,custom_objects={'r2_keras':r2_keras})
Y_pred = model.predict(X)
zero_loc = np.where(Y_pred<0)[0]
Y_pred[zero_loc] = 0

error = np.sqrt((Y_pred-Y)**2)


Stack_Actual = data_array.copy()
Stack_Predict = data_array.copy()
Stack_Predict[:,Y_Loc] = Y_pred.ravel()
#inverse from normalizaton
Data_Array_Actual_invnorm = s.inverse_transform(Stack_Actual)
Data_Array_Predict_invnorm = s.inverse_transform(Stack_Predict)
#Store Y data
Y_pred_ppb = Data_Array_Predict_invnorm[:,Y_Loc]*1000
Y_act_ppb = Data_Array_Actual_invnorm[:,Y_Loc]*1000

#calcualte error
error_ppb = np.sqrt((Y_pred_ppb-Y_act_ppb)**2).reshape(-1,1)
max_error_ppb = np.max(error_ppb)
min_error_ppb = np.min(error_ppb)

#
for i in range(0,len(Sensor_name),1):
    sensor_index = indexall(Sensor_List,Sensor_name[i])
    #plot datetime vs abs error
    fig = plt.figure()
    plt.plot(date[sensor_index],\
                error[sensor_index],'-k',linewidth=2.0)
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.title('Time Series of Sensor:'+Sensor_name[i])
    plt.tight_layout(pad = 0.1)
    name = 'Time_Series'+Sensor_name[i]
    export_graphs(name,fig,filetype='.jpg')

    #plot datetime vs error in ppm

    fig = plt.figure()
    plt.plot(date[sensor_index],\
                error_ppb[sensor_index],'-k',linewidth=2.0)
    plt.xlabel('Date')
    plt.ylabel('Error ($ppb$)')
    plt.title('Real Variable Time Series of Sensor:'+Sensor_name[i])
    plt.tight_layout(pad = 0.1)
    name = 'Real_Variable_Time_Series'+Sensor_name[i]
    export_graphs(name,fig,filetype='.jpg')
    
    #plot datetime vs Ozone in ppm

    fig = plt.figure()
    plt.plot(date[sensor_index],\
                Y_act_ppb[sensor_index],'k',linewidth=2.0,label = 'Data')
    plt.plot(date[sensor_index],\
            Y_pred_ppb[sensor_index],'--b',linewidth=1.0,label = 'Prediction')
    
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('$O_3$ ($ppb$)')
    plt.title('Real Variable Time Series of Sensor:'+Sensor_name[i])
    plt.tight_layout(pad = 0.1)
    name = 'Real_Variable_Time_Series_actual_data_'+Sensor_name[i]
    export_graphs(name,fig,filetype='.jpg')
    
    
    debug = date[sensor_index]
    print(debug[0])
    print(debug[-1])
