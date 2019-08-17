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
import matplotlib.pyplot as plt
from Analyzer_Functions import export_graphs


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



def output_graphs(output_dim_,Y_header,Y_valid,Y_pred,name,filetype_s = '.jpg',\
                  x_fig = 8,y_fig = 3,color = ['r','g','b'],size = 0.05,\
                  parity=True,limits=True,\
                  xlimit = np.array([-0.1,1.1]),ylimit = np.array([-0.1,1.1])):
    
    xy_parity = np.linspace(0,1,100)
    fig = plt.figure(figsize=(x_fig,y_fig))
    for i in range(0,output_dim_,1):
        plt.subplot(1, output_dim_, i+1)
        plt.title(Y_header[i])
        plt.scatter(Y_valid[:,i],Y_pred[:,i],label = Y_header[i],c = color[i],s = size)
        if parity:    
            plt.plot(xy_parity,xy_parity,'k')
        if limits:
            plt.xlim(xlimit[0],xlimit[1])
            plt.ylim(ylimit[0],ylimit[1])
        plt.tight_layout()
    export_graphs(name,fig,filetype = filetype_s)


def output_date_graph(output_dim_,Y_header,Date_array,Y,Y_pred,name,
                      fsize = [10,8],color = ['r','g','b'],size = 1,\
                      filetype_s = '.jpg'):
    fig = plt.figure(figsize=fsize)

    for i in range(0,output_dim_,1):
        plt.subplot(output_dim_,1 , i+1)
        plt.title(Y_header[i])
    
        plt.scatter(Date_array,Y[:,i],c = 'k',s = size)
        plt.scatter(Date_array,Y_pred[:,i],c = color[i],s = size)
        plt.tight_layout()
    export_graphs(name,fig,filetype = filetype_s)

def output_error_graph(output_dim_,Y_header,Date_array,Y,Y_pred,name,
                      fsize = [10,8],color = ['r','g','b'],size = 1,\
                      filetype_s = '.jpg'):
    fig = plt.figure(figsize=fsize)
    error = (Y_pred-Y)**2
    for i in range(0,output_dim_,1):
        plt.subplot(output_dim_,1 , i+1)
        plt.title(Y_header[i])
        
        plt.scatter(Date_array,error[:,i],c = color[i],s = size)
        plt.tight_layout()
    export_graphs(name,fig,filetype = filetype_s)

