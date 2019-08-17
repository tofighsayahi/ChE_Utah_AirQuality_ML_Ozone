# -*- coding: utf-8 -*-
"""
Title : Normalization Test Script

Script used for testing if normalization was done correctly
Created on Mon Aug  5 11:47:42 2019

@author: Tim
"""
#Sepecify these few things and the whole thing will test
number_random = 100
all_data_path =\
r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\PM10\All_Data_norm.csv"
train_valid_path =\
r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\PM10\train_validate_data.csv"



import numpy as np
import pandas as pd
from statistics import mode 
#function that deletes unnecessary stuff
def clean_dataframe_list(df,delete_list):
    header = list(df)
    for i in range(0,len(delete_list)):
      index = header.index(delete_list[i])
      del df[header[index]]
      del header[index]
    return df 
#load data in
df_data_all = pd.read_csv(all_data_path)
df_valid_train = pd.read_csv(train_valid_path)

header_all = list(df_data_all)


#delete the list of variables we dont care about
#delete_list = ['date','Sensor']
#df_valid_train = clean_dataframe_list(df_valid_train,delete_list)
header_vt = list(df_valid_train)

#get different random locations
shapes = df_valid_train.shape
#sets up storage
random_indexes =\
np.random.randint(0,shapes[0],number_random).reshape(-1,1)
fail_headers = []
pass_fail = np.zeros(number_random,dtype = bool)
value_save = np.zeros(number_random)
for i in range(0,number_random,1):
    #ranodmely checks certain rows and finds a match
    temp_train = df_valid_train.loc[random_indexes[i]]
    header_temp = list(temp_train)
    temp_store = []
    for j in range(0,len(header_temp),1):
        value_temp = np.array(temp_train[header_temp[j]])
        
        array_check_temp =\
        np.array(df_data_all[header_temp[j]])
        error = np.abs(value_temp-array_check_temp)
        temp_store+=list(np.where(error<1e-3)[0])
    common_values = mode(temp_store)
    values = temp_store.count(common_values)/(len(header_temp))
    #finds why and where it fails exactly
    if values==1:
        pass_fail[i] = True
        fail_headers.append('Pass')
    else:
        debug_header = []
        for j in range(0,len(header_temp),1):
            value_temp = np.array(temp_train[header_temp[j]])
            
            array_check_temp =\
            np.array(df_data_all[header_temp[j]])
            
            logic_list = list(np.where(value_temp==array_check_temp)[0])
            occur_header = logic_list.count(common_values)
            if occur_header==0:
                debug_header.append(header_temp[j])
        fail_headers.append(debug_header)

    #saves value
    value_save[i] = common_values
    