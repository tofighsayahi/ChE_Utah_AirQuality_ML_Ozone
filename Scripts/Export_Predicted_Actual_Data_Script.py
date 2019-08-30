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

export_data_path =\
r"E:\PhD project\ozone\08212019_All_Data_End"

export_actual_name =\
'All_Data.csv'

export_pred_name =\
'All_Data_pred.csv'

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
zero_loc = np.where(Y_pred<0)[0]
Y_pred[zero_loc] = 0
#restack data
Stack_Actual = data_array.copy()
Stack_Predict = data_array.copy()
Stack_Predict[:,Y_Loc] = Y_pred.ravel()
#use normalization function to get out of normalization
Data_Array_Actual_invnorm = s.inverse_transform(Stack_Actual)
Data_Array_Predict_invnorm = s.inverse_transform(Stack_Predict)

#get data together and export 
df_act = pd.DataFrame(data = Data_Array_Actual_invnorm,columns = header)
df_act = pd.concat([df_omit,df_act],axis=1,join='outer')
path_actual = os.path.join(export_data_path,export_actual_name)
df_act.to_csv(path_actual,index=False)


df_pred = pd.DataFrame(data = Data_Array_Predict_invnorm,columns = header)
df_pred = pd.concat([df_omit,df_pred],axis=1,join='outer')
path_pred = os.path.join(export_data_path,export_pred_name)
df_pred.to_csv(path_pred,index=False)

