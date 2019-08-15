# -*- coding: utf-8 -*-
"""
Title : Data Split Script

Description : This is a script that is self contained that splits the "other dataset"
from the training/validation dataset

Created on Tue Aug 03 21:36:12 2019
Revised and Commented 08/09/2019
@author: Timothy Quah
"""

import pandas as pd
import os

#List of all files that are imported could be one or multiple files
import_list_files =\
[r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\PM1\All_Data_norm.csv",\
 r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\PM1_SWD_PM10\All_Data_norm.csv",\
 r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\PM10\All_Data_norm.csv",\
 r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\SWD\All_Data_norm.csv"]

#List of all the directories to output to should be the same length as import_list_files
export_list =\
[r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\PM1",\
 r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\PM1_SWD_PM10",\
 r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\PM10",\
 r"D:\AirQuality _Research\Data\Full_Redo_Data_Normalization\Remove_Params\SWD"]


#names of export file names
name_list = ['outsider_data.csv','train_validate_data.csv',\
             'outsider_data_extra.csv','train_validate_data_extra.csv']
#variables that should be omited
omit_list = ['date','Sensor']

for i in range(0,len(import_list_files),1):
    df = pd.read_csv(import_list_files[i])
    df_omit = df[omit_list]
    #check other list by the omit if "ts" is in the name it will be put in the other dataset
    Sensor_List =df[['Sensor']].values.T.tolist()[0]
    dates = df['date'].copy()
    unique_sensor = list(set(Sensor_List))
    #removes omits
    for j in range(0,len(omit_list),1):
        del df[omit_list[j]]
    other_index = []
    valid_index = []
    #if 'ts' in the name then put in the other list
    for j in range(0,len(Sensor_List),1):
        if Sensor_List[j][0:2]=='ts':
            other_index.append(j)
        else:
            valid_index.append(j)
    #get other and valid datasets
    df_other = df.loc[other_index]
    df_valid = df.loc[valid_index]
    
    df_other_extra = pd.concat([df_omit.loc[other_index],df_other],\
                               axis = 1,join = 'outer')
    
    df_valid_extra = pd.concat([df_omit.loc[valid_index],df_valid],\
                               axis = 1,join = 'outer')
    
    export_list_df = [df_other,df_valid,df_other_extra,df_valid_extra]
    #export the 2 csvs one for the valid/train and one for other
    for j in range(0,len(name_list),1):
        full_export_path = os.path.join(export_list[i],name_list[j])
        export_list_df[j].to_csv(full_export_path,index=False)

