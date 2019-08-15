# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:03:01 2019

@author: Tim
"""
import pandas as pd
import os 
import numpy as np
# fix random seed for reproducibility
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
from Normalizer_Functions import extract_date,clean_dataframe,replace_header_id


def norm_divider(Data_Array,train_percent=0.7,Min = 0,Max = 1,rnum = 7):
    random.seed(rnum)
    np.random.seed(rnum)
    s = MinMaxScaler(feature_range=(Min,Max))
    Data_Array_norm = s.fit_transform(Data_Array)
    data_len = len(Data_Array[:,0])
    train_len = int(np.round(data_len*train_percent))
    scramble_array = np.array(random.sample(range(data_len), data_len))
    train_list = scramble_array[0:train_len]
    valid_list = scramble_array[train_len:]
    return Data_Array_norm,train_list,valid_list

def norm_divider_export_s(Data_Array,Min = 0,Max = 1,rnum = 7):
    random.seed(rnum)
    np.random.seed(rnum)
    s = MinMaxScaler(feature_range=(Min,Max))
    Data_Array_norm = s.fit_transform(Data_Array)
    return Data_Array_norm,s

def normalize_all_folder(s,importpath,exportpath,delete_list,date_exp=True):
    load_data_list = os.listdir(importpath)
    for i in range(0,len(load_data_list),1):
        full_import_path = os.path.join(importpath,load_data_list[i])
        df = pd.read_csv(full_import_path)
        df['date'] = pd.to_datetime(df['date'])
        df_date = extract_date(df)
        df_comb = pd.concat([df_date,df],axis = 1,join = 'outer')
        sensor = 'AIR_U_Sensor'
        header = list(df_comb)
        replace_id = header[4][-4:]        
        delete_new = replace_header_id(delete_list,sensor,replace_id)

        df_clean = clean_dataframe(df_comb,delete_new)
        date = df['date'].copy()
        
        header = list(df_clean)
        data_array = np.array(df_clean)
        data_Array_norm = s.fit_transform(data_array)
        df_norm = pd.DataFrame(data_Array_norm,columns = header)
        if date_exp:
            df_norm = pd.concat([date,df_norm],axis = 1,join = 'outer')
        header = list(df_norm)
        name = load_data_list[i][:-4]+'_norm.csv'
        path = os.path.join(exportpath,name)
#        del df_norm['Unnamed: 0']    
        df_norm.to_csv(path,index=False)        

def import_normalization(import_path =\
                  r"D:\AirQuality _Research\Normalize_models\large_data_norm.pkl"):
    return joblib.load(import_path) 
