# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:26:13 2019

@author: Tim
"""

import numpy as np
from sklearn.externals import joblib
import pandas as pd
from keras.models import load_model
import os
function_path = "../Functions"
os.chdir(function_path)
from Trainer_Functions import r2_keras,model_neural_network,load_evaluate_neural_net,norm_divider,divider_XY,mse,r2
from Analyzer_Functions import export_graphs,plot_settings,plot_parity
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.close('all')

#define all the paths
outside_data_path =\
r"E:\PhD project\ozone\08212019_All_Data_Split\outsider_data.csv"

train_valid_data_path =\
r"E:\PhD project\ozone\08212019_All_Data_Split\train_validate_data.csv"

neural_network_path =\
r"E:\PhD project\ozone\Saved_Neural_Networks\082119_nnet_node_mse_40layer_1new.h5"

norm_path =\
r"E:\PhD project\ozone\08212019_All_Data_Norm\large_data_norm.pkl"

# plot settings set for font sizes
plot_settings(25,22,22,20,[10,10])

#load neural network models
model = load_model(neural_network_path,custom_objects={'r2_keras':r2_keras})
s = joblib.load(norm_path)
#import the train valid data
df = pd.read_csv(train_valid_data_path)
header =list(df) 
header_num = len(header)
Full_List = list(np.arange(0,header_num-1+1e-6,1,dtype=int))
Y_Loc =  header.index('O3 Value')
Y_header_list = []
Y_header_list.append(Y_Loc)
X_header_list = list(set(Full_List)-set(Y_header_list))
data_array = np.array(df)
train_list,valid_list = norm_divider(data_array)
X,Y,X_valid,Y_valid = divider_XY(X_header_list,Y_header_list,data_array,train_list,valid_list)
#try to predict values
Y_pred = model.predict(X_valid)
zero_loc = np.where(Y_pred<0)[0]
Y_pred[zero_loc] = 0

#stack them back into where they came from so that we can reverse the normalization
col = np.shape(data_array)[1]
row = np.shape(X_valid)[0]
Stack_Actual = np.zeros([row,col])
Stack_Predict = np.zeros_like(Stack_Actual)
Stack_Actual[:,X_header_list] = X_valid
Stack_Actual[:,Y_header_list] = Y_valid
Stack_Predict[:,X_header_list] = X_valid
Stack_Predict[:,Y_header_list] = Y_pred
#Inverse both the predicted and the actual data 
Data_Array_Actual_invnorm = s.inverse_transform(Stack_Actual)
Data_Array_Predict_invnorm = s.inverse_transform(Stack_Predict)





#Store Y data
Y_pred_ppm = Data_Array_Predict_invnorm[:,Y_Loc]*1000
Y_act_ppm = Data_Array_Actual_invnorm[:,Y_Loc]*1000


#plot the parity plot for the normalized validation data
fig = plot_parity(Y_valid[:,0],Y_pred[:,0],'Normalized $O_3$ Values','Predicted $O_3$ Values',s = 0.5)
export_graphs('norm_parity_valid',fig,filetype = '.jpg')

#plot the parity plot for the unnormalized validation data
fig = plot_parity(Y_act_ppm,Y_pred_ppm,'Actual $O_3$ Values($ppb$)','Predicted $O_3$ Values($ppb$)',s = 0.5,maxset=True)
export_graphs('unnorm_parity_valid',fig,filetype = '.jpg')

#load in the outside dataset
df_test = pd.read_csv(outside_data_path)
data_array_test = np.array(df_test)
X_Other = data_array_test[:,X_header_list]
Y_Other = data_array_test[:,Y_header_list]
Y_other_pred = model.predict(X_Other)
zero_loc = np.where(Y_other_pred<0)[0]
Y_other_pred[zero_loc] = 0

#stack them back into where they came from so that we can reverse the normalization
col = np.shape(data_array_test)[1]
row = np.shape(X_Other)[0]
Stack_Actual = np.zeros([row,col])
Stack_Predict = np.zeros_like(Stack_Actual)
Stack_Actual[:,X_header_list] = X_Other
Stack_Actual[:,Y_header_list] = Y_Other
Stack_Predict[:,X_header_list] = X_Other
Stack_Predict[:,Y_header_list] = Y_other_pred
Data_Array_Actual_invnorm = s.inverse_transform(Stack_Actual)
Data_Array_Predict_invnorm = s.inverse_transform(Stack_Predict)
#Store Y data
Y_pred_ppm = Data_Array_Predict_invnorm[:,Y_Loc]*1000
Y_act_ppm = Data_Array_Actual_invnorm[:,Y_Loc]*1000
#plot the parity plot for the normalized other data
fig = plot_parity(Y_Other[:,0],Y_other_pred[:,0],'Normalized $O_3$ Values','Predicted $O_3$ Values',s = 0.5)
export_graphs('norm_parity_other',fig,filetype = '.jpg')
#plot the parity plot for the unnormalized other data
fig = plot_parity(Y_act_ppm,Y_pred_ppm,'Actual $O_3$ Values($ppb$)','Predicted $O_3$ Values($ppb$)',s = 0.5,maxset=True)
export_graphs('unnorm_parity_other',fig,filetype = '.jpg')



