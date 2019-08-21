# -*- coding: utf-8 -*-
"""
Title : Data Normalization Script

Description : This is a script that was uses Normalizer Functions to 
1) Combine the sensor data into one file
2) Remove unwanted variables
3) Return a csv with all the data and a csv with all the data normalized
4) Save and Export the normalization factor

Created on Tue Jun 18 21:36:12 2019
Revised and Commented 08/09/2019
@author: Timothy Quah
"""
import os 
#gets program to look in the directory 
function_path = "../Functions" #path to Tim's function
os.chdir(function_path)
from Normalizer_Functions import Combine_All_Data,clean_dataframe,normalization,\
export_normalization
import pandas as pd

#export combined data file
export_alldata_path =\
r"E:\PhD project\ozone\08212019_All_Data_Norm"

#Import sensor data files 
import_alldata_path =\
r"E:\PhD project\ozone\08212019_All_Data_Clean"

#export name
export_all_data = 'All_Data.csv'

#returns dataframe that has all_data
all_data = Combine_All_Data(import_alldata_path,export_alldata_path,\
                            export_all_data,export_lg = True,minmax = True,\
                            mics=True)
#list of variables that should be omitted from normalization
Omit_list = ['date','Sensor']

#delete from dataframe
delete_list = ['NOX Value','NO Value','MC Value']
delete_list+=['CO Value','NO2 Value']
delete_list+=[ 'SWD Value']
#delete_list+=['PM10_AIR_U_Sensor','Min_PM10_AIR_U_Sensor','Max_PM10_AIR_U_Sensor']
#delete_list+=['PM1_AIR_U_Sensor','Min_PM1_AIR_U_Sensor','Max_PM1_AIR_U_Sensor']
delete_list+=Omit_list

#clean the dataframe yields a dataframe with variables that will be normalized
df_clean = clean_dataframe(all_data,delete_list)

#save omitted values
df_omit = all_data[Omit_list]

#export the normalized 
export_norm_data = 'All_Data_norm.csv'
#get the normalized dataframe and the normalization function
df_norm,s = normalization(df_clean)

#exports the normalized all data 
#add the omitted data
df_comb = pd.concat([df_omit,df_norm],axis = 1,join = 'outer')
full_export_path = os.path.join(export_alldata_path,export_norm_data)
#export to csv
df_comb.to_csv(full_export_path,index=False)

# export the normalization function
export_name = 'large_data_norm.pkl'
export_normalization(s,export_alldata_path,export_name)