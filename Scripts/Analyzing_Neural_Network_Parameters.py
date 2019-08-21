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
from Analyzer_Functions import export_graphs,Parameter_Analysis
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#Setup system
plt.close('all')
random.seed(7)
np.random.seed(7)
#set path to data
full_path =\
r"E:\PhD project\ozone\08212019_All_Data_Split\train_validate_data.csv"

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
r"E:\PhD project\ozone\Saved_Neural_Networks\082119_nnet_node_mse_40layer_1new.h5"
model = load_model(load_model_path,custom_objects={'r2_keras':r2_keras})
model.summary()
#call custom function
fig_save,plt_name_save,range_save = Parameter_Analysis(model,header,X_header_list)
#export figure
for i in range(0,len(fig_save)):
    export_graphs(plt_name_save[i],fig_save[i])



#plot hist
#X_header = header.remove('O3 Value')
#plt.figure()
#plt.bar(range_save,label=X_header)

