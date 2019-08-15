# -*- coding: utf-8 -*-
"""

Title : Data Loading Script 

Description : This is a script that was uses Cleaner_Loader_Functions to 
1) Seperate the raw data from DAQ and AirU datasets
2) Remove Null data from DAQ
3) Add missing data to datasets (DAQ)
4) Cleans Air U dataset 
5) Combines DAQ and AirU datasets
6) Final Data Ordering (Sorts headers by alphabetical order)
7) Export Data
    
Created on Tue Jun 18 21:36:12 2019
Revised and Commented 08/07/2019
@author: Timothy Quah
"""
## Import programs
import os #used to for operating system functionalities
    
## Gets program to look in the directory 
function_path = "../Functions" #path to Tim's function

## Change directory to function directory allows script to use functions
os.chdir(function_path)

## Load Custom functions by Tim 
from Cleaner_Loader_Functions import Main_Sep_Load,DAQ_Parser_Seperator,\
AIR_U_Sensor_Sep,Matchin_DAQ_AIR_U_time,Organize_Clean_All,Export_CSV_ALL,\
null_code_DAQ_filter,Add_Missing_Data,find_in_list,Reorder_df

## The following code is used to extract all the files from AIrU Directories
## Where is the AirU files stored and where do want the script to put them? 

## Import all raw data this is a directory with all of the data both DAQ and AirU
import_path =\
r"D:\AirQuality _Research\Data\Full_Year"
## Export the cleaned and organized csvs from this script
export_path = r"D:\AirQuality _Research\Data\Output_Clean"

## Locations Usually Hawthorne and RosePark
## Must put location with solar radiation first
Location_name = ['Hawthorne','Rose']
Location_abv = ['HW','RP']

## Loads the data from the script
df_DAQ_dict,df_AirU_dict = Main_Sep_Load(import_path,Location_name,Location_abv)

#This is to create pure "other" set 
air_dict_list = list(df_AirU_dict)

for i in range(0,len(air_dict_list),1):
    new_id = 'ts'+Location_abv[i] #replace ID
    header_air = list(df_AirU_dict[air_dict_list[i]]) #get current header
    header_change = find_in_list(header_air,'.1') # finds duplicates
    for j in range(0,len(header_change)):
        new_header = header_change[j][:-6]+new_id #changes id
        #saves and adds the duplicates into the dataset
        df_AirU_dict[Location_name[i]] = df_AirU_dict[Location_name[i]].rename(columns = {header_change[j]:new_header}) 


#seperates DAQ dataset to Data,Flags,Nullcode
df_DAQ_values_dict_unfilter,df_DAQ_Flags_dict,df_DAQ_Null_Code_dict,\
            Symbol_Flag_list_unique,\
            Symbol_Null_Code_list_unique =\
            DAQ_Parser_Seperator(df_DAQ_dict,Location_name,Location_abv)

#Adds missing values to DAQ-in the case it was written for rose park
df_DAQ_values_dict_miss,df_DAQ_NC_dict_miss =\
 Add_Missing_Data(df_DAQ_values_dict_unfilter,df_DAQ_Null_Code_dict,'SR Value')

#removes Turns null code to NaN values in DAQ dataset and deletes rows
df_DAQ_values_dict = null_code_DAQ_filter(df_DAQ_NC_dict_miss,\
                                          df_DAQ_values_dict_miss)
      
# delete dictionary for memory purposes 
del df_DAQ_dict


#cleans up AirU data ie removes Nan values
df_AirU_sensor_dict = AIR_U_Sensor_Sep(df_AirU_dict)

#Matches AirU and DAQ dataset and removes other values
df_All_Data_dict = Matchin_DAQ_AIR_U_time(df_AirU_sensor_dict,df_DAQ_values_dict)

#Final Cleaning of data NaNs
df_All_Data_dict_clean = Organize_Clean_All(df_All_Data_dict)

#Final Data Ordering
df_Reorder_Data_dict = Reorder_df(df_All_Data_dict_clean)

#Exports CSVs
Export_CSV_ALL(df_Reorder_Data_dict,export_path)
        
    

