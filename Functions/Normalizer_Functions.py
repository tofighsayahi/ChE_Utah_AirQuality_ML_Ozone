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

import pandas as pd
import os 
import numpy as np
# fix random seed for reproducibility
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
from Cleaner_Loader_Functions import indexall,find_in_list
from copy import deepcopy
#from Normalizer_Functions import Combine_All_Data,clean_dataframe,normalization,\
#export_normalization

"""
Title:
extract_date

Description: Used to extract the hour of day/day of the week/month of the year
    
Function Dependencies:
Python 3
pandas

Inputs:
df - all data dataframe dtype = dict

Optional Inputs:
None 

Outputs:
df_date - outputs a dataframe that has date/hour/month

Optional Outputs:
None 
"""

def extract_date(df):
    #lists
    hour_of_day = []
    day_of_week = []
    month_of_year = []
    for i in range(0,len(df['date']),1):
        #extract values
        hour_of_day.append(df['date'][i].hour)
        day_of_week.append(df['date'][i].weekday())
        month_of_year.append(df['date'][i].month)
    #put dataframe together
    df_date = pd.DataFrame({'Day':day_of_week,'Hour':hour_of_day,\
                            'Month':month_of_year})
    #output dataframe
    return df_date

"""
Title:
replace_header_id

Description: replace sensor id and replace with a generic name 
    
Function Dependencies:
Python 3
find_in_list
indexall

Inputs:
header- list of headers dtype = list
sensor_id-the sensor id dtype = str

Optional Inputs:
replace_id- what to replace the sensor id with dtype = str

Outputs:
header_new-the new header stripped of the sensor id and replaced with a generic name

Optional Outputs:
None 
"""

def replace_header_id(header,sensor_id,replace_id = 'AIR_U_Sensor'):
    #find sensor ids in header
    replace_header = find_in_list(header,sensor_id)
    header_new = deepcopy(header) 
    for i in range(0,len(replace_header),1):
        #find where each sensor header is
        replace_index = indexall(header,replace_header[i])[0]
        replace_length = len(sensor_id)
        #replace header names
        header_new[replace_index] = header[replace_index][:-replace_length]+replace_id
    return header_new

"""
Title:
Combine_All_Data

Description: Combines all the data, adds min and max columns, and export a dataframe
with all the data
    
Function Dependencies:
Python 3
os
numpy
replace_header_id
extract_date

Inputs:
import_path - the path to the clean files from data cleaner script dtype = str
export_path - the path to export the all data csv dtype = str
export_name - name of the csv dtype = str

Optional Inputs:
export_lg - export is an option dtype = bool
date_convert- extract date is an option dtype = bool
minmax - minmax addition is an option dtype = bool
minmax_num - number to average to make min and max dtype = int

Outputs:
df_comb- dataframe with all the data with additional additions

Optional Outputs:
None 
"""



def Combine_All_Data(import_path,export_path,export_name,\
                     export_lg = True,date_convert=True,minmax = False,\
                     minmax_num = 5,mics = False,time_offset = -7,time_mics =\
                     np.array([14,22])):
    #load the data in
    load_data_list = os.listdir(import_path)
    sensor_list = []
    for i in range(0,len(load_data_list)):
        full_import_path = os.path.join(import_path,load_data_list[i])
        df = pd.read_csv(full_import_path)
        #if you include the date convert
        if date_convert:
            df['date'] = pd.to_datetime(df['date'])
            df_date = extract_date(df)
            #add it to the dataframe
            df = pd.concat([df_date,df],axis = 1,join = 'outer')
            df_nn = df.copy()
            header_nn = list(df_nn)

            sensor_id = header_nn[4][-4:]
#            print(sensor_id)

        else:
            df_nn = df.copy()
            header_nn = list(df_nn)
            sensor_id = header_nn[6][-4:]
        #gets sensor id
        Data_Array_temp = np.array(df_nn)
        repeat = np.shape(Data_Array_temp)[0]
        for j in range(0,repeat,1):
            sensor_list.append(sensor_id)
        #stack each dataframe on top of one another
        if i==0:
            Data_Array = Data_Array_temp.copy()
        else:
            Data_Array = np.vstack([Data_Array,Data_Array_temp])
        #if minmax take the average and make the lower average min and upper average 
        #This is done per sensor
        if minmax:
            header_sensor = find_in_list(header_nn,sensor_id)
            for j in range(0,len(header_sensor),1):
                index = indexall(header_nn,header_sensor[j])[0]
                sort_array = np.sort(Data_Array_temp[:,index])
                min_value = np.mean(sort_array[:minmax_num])*np.ones_like(sort_array)
                max_value = np.mean(sort_array[-minmax_num:])*np.ones_like(sort_array)
                if j==0:
                    Add_Data_temp = np.column_stack((min_value,max_value))
                else:
                    Add_Data_temp = np.column_stack((Add_Data_temp,min_value,max_value))
            if i==0:
                Add_Data = Add_Data_temp.copy()
            else:
                Add_Data = np.vstack([Add_Data,Add_Data_temp])
    #add this to the a data frame and make sure it is added to the headers    
    #add mics to dataframe
    if mics:
        temp_header = 'MICS '+sensor_id
        header_nn.append(temp_header)
        Hour_loc = indexall(header_nn,'Hour')[0]
        Hour_Array = np.array(Data_Array[:,Hour_loc])
        Mics_Array = np.zeros_like(Hour_Array)
        for j in range(0,len(time_mics),1):
            adj_time_mics = np.mod(time_mics[j]+time_offset,24)
            loc = np.where(Hour_Array==adj_time_mics)[0]
            Mics_Array[loc] = 0.5
        Data_Array = np.column_stack((Data_Array,Mics_Array))
        mics_min = np.zeros_like(Mics_Array)
        mics_max = np.ones_like(Mics_Array)*0.5
        Add_Data = np.column_stack((Add_Data,mics_min))
        Add_Data = np.column_stack((Add_Data,mics_max))

            
    if minmax:
        Data_Array = np.column_stack((Data_Array,Add_Data))
        header_sensor = find_in_list(header_nn,sensor_id)
        header_add = []
        for j in range(0,len(header_sensor),1):
            header_add.append('Min_'+header_sensor[j])
            header_add.append('Max_'+header_sensor[j])
        header_nn+=header_add

    #replace the header id
    header_new = replace_header_id(header_nn,sensor_id)
    print(sensor_id)
    #make the new dataframe and combine everything
    df_all_data = pd.DataFrame(data = Data_Array,columns = header_new)
    df_sensor = pd.DataFrame(sensor_list, columns=['Sensor'])    
    df_comb = pd.concat([df_sensor,df_all_data],axis = 1,join = 'outer')
    #export if desired
    if export_lg:
        full_export_path = os.path.join(export_path,export_name)
        df_comb.to_csv(full_export_path,index=False)
    return df_comb
    
"""
Title:
clean_dataframe

Description: Removes unwanted or unneeded variables
    
Function Dependencies:
Python 3

Inputs:
df- dataframe that is inputed dtype = dataframe
delete_list - delete the list of variables dtype = list

Optional Inputs:

Outputs:
df_data_pre_norm - dataframe without the unwanted/uneeded variables dtype = dataframe

Optional Outputs:
None 
"""

def clean_dataframe(df,delete_list):
    #copy dataframe
    df_data_pre_norm = df.copy()
    for i in range(0,len(delete_list)):
        #delete the columns
        del df_data_pre_norm[delete_list[i]]
    return df_data_pre_norm

"""
Title:
normalization

Description: Normalizes a dataset and returns the normalized dataset and the normalizing function
    
Function Dependencies:
Python 3
sklearn
pandas 
numpy 
random 

Inputs:
df - dataframe of normalizable data NO DATES OR STRINGS dtype = dataframe

Optional Inputs:
Min - Minimum value dtype = float
Max - Maximum value dtype = float
rnum -random seed  dtype = int

Outputs:
df_norm - normalized dataframe returned dtype = dataframe
s - normalizing function can be saved an used to unormalize the data dtype = object

Optional Outputs:
None 
"""

def normalization(df,Min = 0,Max = 1,rnum = 7):
    #random seed (idk if useful)
    random.seed(rnum)
    np.random.seed(rnum)
    # data array
    data_array = np.array(df)
    #setup normalization
    s = MinMaxScaler(feature_range=(Min,Max))
    #normalize data
    data_Array_norm = s.fit_transform(data_array)
    header = list(df)
    #put back together for dataframe
    df_norm = pd.DataFrame(data_Array_norm,columns = header)
    return df_norm,s

"""
Title:
export_normalization

Description: Does exactly what it says exports the normalization tool
    
Function Dependencies:
Python 3
sklearn

Inputs:
s - normalizing function can be saved an used to unormalize the data dtype = object
export_path - path to export to dtype = str
name - name that the normalizing function is to be saved as  dtype = str
Optional Inputs:
replace_id- what to replace the sensor id with dtype = str

Outputs:
None 

Optional Outputs:
None 
"""
def export_normalization(s,export_path,name):
    #get full name
    full_export_path = os.path.join(export_path,name)
    # saves normalization
    joblib.dump(s,full_export_path)

