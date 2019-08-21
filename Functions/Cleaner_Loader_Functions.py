# -*- coding: utf-8 -*-
"""

Title : Cleaner_Loader_Functions

Description : This the functions that allow Data_Loading_Script to Work
1) Separate the raw data from DAQ and AirU datasets
2) Remove Null data from DAQ
3) Add missing data to datasets (DAQ)
4) Cleans Air U dataset 
5) Combines DAQ and AirU datasets
6) Final Data Ordering (Sorts headers by alphabetical order)
7) Export Data

Function Dependencies:
panda 
numpy 
os 
copy
    
Created on Tue Jun 18 21:36:12 2019
Revised and Commented 08/08/2019
@author: Timothy Quah
"""
#these are the packages/tools we need import
import os 
import pandas as pd
import numpy as np
from copy import deepcopy #super important to deepcopy things if you are not sure



"""
Title:
indexall

Description:
Used to get index of value in a list. Example pass lst =[0,1,2] and value = 0 

Function Dependencies:
Python 3
    
Inputs:
lst can be a list of int/float/str
float can be a single int/float/str

Optional Inputs:
None

Outputs:
index of where value is in list is an int    

Optional Outputs:
None 

Example usage:

Input:
    
lst = [0,1,2]
value = 0
index = indexall(lst, value)

Ouput:
index = 0 

"""
def indexall(lst, value):
    return [i for i, v in enumerate(lst) if v == value]

"""
Title:
find_in_list

Description:
Used to match strings 

Function Dependencies:
Python 3
    
Inputs:
lst can be a list of str
value can be a single str

Optional Inputs:
None

Outputs:
Is a list of strings in the list that match the value

Optional Outputs:
None 

Example usage:

Input:
    
lst = ['apple','oranges']
value = 'app'
a = find_in_list(lst, value)

Ouput:

a = ['apple']
"""
def find_in_list(lst,value):
    return [s for s in lst if value in s]

"""
Title:
sub_main_merge

Description:
Used to move files in subdirectories to a single directory

Function Dependencies:
Python 3
os
shutil
    
Inputs:
import_path is a path to the main folder datatype is a string
export_path is a path to the folder that one wants to export datatype is a string

Optional Inputs:
None

Outputs:
None

Optional Outputs:
None 
"""

def sub_main_merge(import_path,export_path):
    import os
    import shutil
    for root, dirs, files in os.walk(import_path):  # replace the . with your starting directory
       for file in files:
          path_file = os.path.join(root,file)
          shutil.copy2(path_file,export_path) # change you destination dir
    print('Completed')

"""
Title:
DAQ_DATA_Filter

Description:
Used in Main_Sep_Load to Separate DAQ dataset

Function Dependencies:
Python 3
os
pandas

Inputs:
df_DAQ-dataframe from DAQ dtype=dataframe
Location_abv-List of location abbreviations dytype=list

Optional Inputs:
None

Outputs:
df_DAQ_dict-dictionary of dataframes DAQ dtype = dict

Optional Outputs:
None 
"""



def DAQ_DATA_Filter(df_DAQ,Location_abv):
    DAQ_header = list(df_DAQ) # this is how you get headers out
    location_loc = [] #setup list
    for i in range(0,len(Location_abv),1):
        temp_loc= [s for s in DAQ_header if Location_abv[i] in s] #get location
        location_loc.append(temp_loc) # append to list
        
    df_DAQ_dict = dict() #setup dictionary
    for i in range(0,len(Location_abv),1): #itterate over all headers
        for j in range(0,len(location_loc[i]),1): # split between locations
            # logic helps put in the right dictionary
            if j==0:
                df_DAQ_dict[Location_abv[i]] = df_DAQ[location_loc[i][j]]
            else:
               df_DAQ_dict[Location_abv[i]] = pd.concat([df_DAQ[location_loc[i][j]],df_DAQ_dict[Location_abv[i]]],\
                           axis=1,join='outer')
               
        #get headers
        header_1 = list(df_DAQ_dict[Location_abv[i]].iloc[0,:])
        header_2 = list(df_DAQ_dict[Location_abv[i]].iloc[1,:])
        main_header = []
        for j in range(0,len(header_1),1):
            main_header.append(header_1[j]+' '+header_2[j])
        #clean up have some wierd additions like indexes
        df_DAQ_dict[Location_abv[i]].columns = main_header
        df_DAQ_dict[Location_abv[i]] = df_DAQ_dict[Location_abv[i]].drop([0,1])
        df_DAQ_dict[Location_abv[i]] = df_DAQ_dict[Location_abv[i]].reset_index()
        df_DAQ_dict[Location_abv[i]] = df_DAQ_dict[Location_abv[i]].drop(columns = ['index'])
    df_DAQ_dict['Date'] = df_DAQ['Date']
    df_DAQ_dict['Date'] = df_DAQ_dict['Date'].drop([0,1])
    df_DAQ_dict['Date'] = df_DAQ_dict['Date'].reset_index()


    return df_DAQ_dict

"""
Title:
Main_Sep_Load

Description:
Used to load datafiles from DAQ and AirU and for DAQ Separates data into locations
for AirU it will Separate  into data into locations and into sensors

Function Dependencies:
Python 3
os
pandas
DAQ_DATA_Filter

Inputs:
import_path-path to directory with all the data dtype=str

Location_keyword-list of locations dtype=list 

Location_abv-list of location abbreviation dtype=list

Optional Inputs:
file_type-type of file loading '.csv' dtype = str
NOTE: Only loads '.csv' files

DAQ_keyword-keyword for DAQ csv files dtype = str

Outputs:
df_DAQ_dict-dictionary of dataframes DAQ dtype = dict
df_AirU_dict-dictionary of dataframes AirU dtype = dict

Optional Outputs:
None 
"""

def Main_Sep_Load(import_path,Location_keyword,Location_abv,file_type = '.csv',\
                  DAQ_keyword = 'DAQ'):
    #get full list of files in directory
    load_data_list = os.listdir(import_path)
    #extract all data and store data frames in dictionary
    Air_U_dict = dict()
    count = np.zeros(len(Location_keyword))
    for i in range(0,len(Location_keyword),1):
        Air_U_dict[Location_keyword[i]] = [] 
    for i in range(0,len(load_data_list),1):
        import_fullpath = os.path.join(import_path,load_data_list[i])
        #checks to make sure its .csv
        if load_data_list[i].endswith(file_type):
            #logic for DAQ files
            #uses DAQ string in file name to different treatment
            if load_data_list[i].find(DAQ_keyword)>=0:
                df = pd.read_csv(import_fullpath)
                #remove the unnecessary index/rows
                df_DAQ_dict = DAQ_DATA_Filter(df,Location_abv)
            #logic for Air U files works for files with same time frame if using  
            #Separate  time frame a different approach should be used
            else:
                # check for each location for AirU
                for j in range(0,len(Location_keyword),1):
                    if load_data_list[i].find(Location_keyword[j])>=0:                        
                        df_temp = pd.read_csv(import_fullpath)
                        if count[j]==1:
                            header = list(df_temp)
                            del df_temp[header[0]]
                        Air_U_dict[Location_keyword[j]].append(df_temp)
                        count[j] = 1
        else:
            print('Function does not recognize files that are not .csv files')
    del df
    del df_temp
    df_AirU_dict = dict()
    for i in range(0,len(Location_keyword),1):
        df_AirU_dict[Location_keyword[i]] =\
        pd.concat(Air_U_dict[Location_keyword[i]], axis=1,join='outer')  
    #returns single dataframe for each location in two dictionaries
    return df_DAQ_dict,df_AirU_dict






def check_logic(l1,l2):
    if len(l1)==len(l2): # check is list1 is the same length as list 2
        print('Logic ok')
    else:
        print('Logic Fail')

"""
Title:
DAQ_Parser_Seperator

Description:
Used to Separate DAQ dataframe values from null code and flags

Function Dependencies:
Python 3
os
pandas
find_in_list

Inputs:
df_DAQ_dict-DAQ dataframe dictionary dtype = dict
Location_name - list of locations dtype = list
Location_abv - list of locations abbreviations dtype = list

Optional Inputs:
None 


Outputs:
df_DAQ_values_dict - DAQ dataframe dictionary with values dtype = dict
df_DAQ_Flags_dict - DAQ dataframe dictionary with flags dtype = dict
df_DAQ_Null_Code_dict - DAQ dataframe dictionary with Null Code dtype = dict
Symbol_Flag_list_unique - List of unique flags dtype = list
Symbol_Null_Code_list_unique - List of unique Null Code dtype = list

Optional Outputs:
None 
"""

def DAQ_Parser_Seperator(df_DAQ_dict,Location_name,Location_abv):
    # Setup dictionaries and list
    df_DAQ_values_dict = dict()
    df_DAQ_Flags_dict = dict()
    df_DAQ_Null_Code_dict = dict()
    Symbol_Flag_list = []
    Symbol_Null_Code_list = []
    #itterate over location 
    for i in range(0,len(Location_name),1):
        #get headers
        header_list_temp = list(df_DAQ_dict[Location_abv[i]])
        #find values/flags/nullcode location
        value_list = find_in_list(header_list_temp, 'Value')
        Flags_list = find_in_list(header_list_temp, 'Flags')
        Null_Code_list = find_in_list(header_list_temp, 'Null Code')
        #get date and put into values/flags/nullcode 
        df_DAQ_values_dict[Location_name[i]] = pd.to_datetime(df_DAQ_dict['Date']['Date'])
        df_DAQ_Flags_dict[Location_name[i]] = pd.to_datetime(df_DAQ_dict['Date']['Date'])
        df_DAQ_Null_Code_dict[Location_name[i]] = pd.to_datetime(df_DAQ_dict['Date']['Date'])
        # Itterate over the variables
        for j in range(0,len(value_list),1):
            #change datatype to float64 from string
            rep = df_DAQ_dict[Location_abv[i]][value_list[j]].astype('float64')
            #Concat values to dataframe-values
            df_DAQ_values_dict[Location_name[i]] = pd.concat([df_DAQ_values_dict[Location_name[i]],\
                               rep],\
                                axis=1,join='outer')
            #Concat values to dataframe-flags
            df_DAQ_Flags_dict[Location_name[i]] = pd.concat([df_DAQ_Flags_dict[Location_name[i]],\
                       df_DAQ_dict[Location_abv[i]][Flags_list[j]]],\
                        axis=1,join='outer')
    
            #get symbol list
            Symbol_Flag_list += list(df_DAQ_Flags_dict[Location_name[i]][Flags_list[j]].unique())
            
            #Concat values to dataframe-null code
            df_DAQ_Null_Code_dict[Location_name[i]] = pd.concat([df_DAQ_Null_Code_dict[Location_name[i]],\
                               df_DAQ_dict[Location_abv[i]][Null_Code_list[j]]],\
                                axis=1,join='outer')
            #get null code list
            Symbol_Null_Code_list  += list(df_DAQ_Null_Code_dict[Location_name[i]][Null_Code_list[j]].unique())
        #get headers 
        header = list(df_DAQ_values_dict[Location_name[i]])
        #set headers over flags
        df_DAQ_Flags_dict[Location_name[i]].columns = header
        #set header over null code
        df_DAQ_Null_Code_dict[Location_name[i]].columns = header
    #get unique flags and nullcodes
    Symbol_Flag_list_unique = list(set(Symbol_Flag_list))
    Symbol_Flag_list_unique.remove(Symbol_Flag_list_unique[0])
    Symbol_Null_Code_list_unique = list(set(Symbol_Null_Code_list))
    Symbol_Null_Code_list_unique.remove(Symbol_Null_Code_list[0])
    # return the values
    return df_DAQ_values_dict,df_DAQ_Flags_dict,df_DAQ_Null_Code_dict,\
            Symbol_Flag_list_unique,Symbol_Null_Code_list_unique

"""
Title:
Organize_Clean_DAQ

Description:
Used to delete/replace missing values if 2 NaN values are next to each other
the entire row of data will be deleted 

Function Dependencies:
Python 3
os
pandas


Inputs:
df_DAQ_values_dict - dictionary of DAQ values dtype = dict

Optional Inputs:
None 


Outputs:
df_DAQ_values_dict - outputs a clean dictionary of DAQ values dtype = dict

Optional Outputs:
None 
"""


def Organize_Clean_DAQ(df_DAQ_values_dict):
    #get location
    Location_name = list(df_DAQ_values_dict)
    #iterate over location
    for j in range(0,len(Location_name),1):
        #get numpy array if Nan or not 
        logic_NAN = np.array(df_DAQ_values_dict[Location_name[j]].isnull())
        column_header = list(df_DAQ_values_dict[Location_name[j]])
        logic_nan_shape = np.shape(logic_NAN)
        #columns_null = np.unique(test2[1],return_counts=True)
        delete_index_list = []
        #find the NAn value
        for i in range(0,logic_nan_shape[0],1):
            sum_row = np.sum(logic_NAN[i,:])
            if sum_row>0:
                #if there is one Nan value in two consecutive the entire row will be scrapped
                up_down_check = np.sum((logic_NAN[i,:]*logic_NAN[i-1,:])+(logic_NAN[i,:]*logic_NAN[i+1,:]))
                if up_down_check>0:
                    delete_index_list.append(i)
                else:
                    patch_loc = np.where(logic_NAN[i,:])[0]
                    for k in range(0,len(patch_loc),1):
                        l = patch_loc[k]
                        replace_value = (df_DAQ_values_dict[Location_name[j]][column_header[l]][i+1]+\
                         df_DAQ_values_dict[Location_name[j]][column_header[l]][i-1])/2
                                         
                        df_DAQ_values_dict[Location_name[j]].at[i,column_header[l]] = replace_value
        #drop and delte index
        df_DAQ_values_dict[Location_name[j]] = df_DAQ_values_dict[Location_name[j]].drop(delete_index_list)
        df_DAQ_values_dict[Location_name[j]] = df_DAQ_values_dict[Location_name[j]].reset_index()
#        del df_DAQ_values_dict[Location_name[j]]['index']
    return df_DAQ_values_dict


"""
Title:
AIR_U_Sensor_Sep

Description:
Separates AirU datasets into specific sensors and converts time zone

Function Dependencies:
Python 3
os
pandas


Inputs:
df_AirU_dict - dictionary of AirU  values dtype = dict

Optional Inputs:
data_offset - convert GST to MST (7 hour difference) dtype = int
dupes - deletes duplicate sensors dtype = bool

Outputs:
df_AirU_sensor_dict - outputs a clean dictionary of AirU values Separate d
by sensor and the date is converted to the correct timezone dtype = dict

Optional Outputs:
None 
"""
def AIR_U_Sensor_Sep(df_AirU_dict,data_offset = -7,dupes = False):
    #Setup new dictionary and get locations
    df_AirU_sensor_dict = dict()
    Location_name = list(df_AirU_dict)
    #itterate over locatiions
    for i in range(0,len(Location_name),1):
        #start putting together new dictionary dataframes
        df_AirU_sensor_dict[Location_name[i]] = dict()
        header = list(df_AirU_dict[Location_name[i]])
        df_AirU_dict[Location_name[i]] = df_AirU_dict[Location_name[i]].rename(columns = {header[0]:'date'})
        header = list(df_AirU_dict[Location_name[i]])
        df_AirU_dict[Location_name[i]][header[0]] = pd.to_datetime(df_AirU_dict[Location_name[i]][header[0]])
        #Change the time
        df_AirU_dict[Location_name[i]][header[0]] = df_AirU_dict[Location_name[i]][header[0]]+pd.DateOffset(hours = data_offset) 
        if dupes:
            for j in range(0,len(header),1):
                #remove duplicates
                if header[j].find('.')>0:
                    del df_AirU_dict[Location_name[i]][header[j]]
        header = list(df_AirU_dict[Location_name[i]])
        #get the sensors
        header_params = header[1:]
        sensor_id = []
        for j in range(0,len(header_params),1):
            #get sensor ids
            sensor_id.append(header_params[j][-4:])
        sensor_id = list(set(sensor_id))
        for j in range(0,len(sensor_id),1):
            #Separate  based on sensor id
            df_AirU_sensor_dict[Location_name[i]][sensor_id[j]] =  df_AirU_dict[Location_name[i]][header[0]]
            for k in range(0,len(header_params),1):
                if header_params[k].find(sensor_id[j])>0:
                    
                    df_AirU_sensor_dict[Location_name[i]][sensor_id[j]] =\
                    pd.concat([df_AirU_sensor_dict[Location_name[i]][sensor_id[j]],\
                               df_AirU_dict[Location_name[i]][header_params[k]]],axis=1,join = 'outer')
    return df_AirU_sensor_dict

"""
Title:
Matchin_DAQ_AIR_U_time

Description:
Matches DAQ and AIR_U datasets
Function Dependencies:
Python 3
os
pandas


Inputs:
df_AirU_sensor_dict-dictionary of AirU dataframes Separated by sensor dtype = dict
df_DAQ_values_dict-dictionary of DAQ dataframes Separated by Location dtype = dict

Optional Inputs:
None 

Outputs:
df_All_Data_dict - dictionary of combined AirU/DAQ dtype = dict

Optional Outputs:
None 
"""

#Matches DAQ and AirU time
def Matchin_DAQ_AIR_U_time(df_AirU_sensor_dict,df_DAQ_values_dict):
    #get location and set dictionary
    Location = list(df_AirU_sensor_dict)
    df_All_Data_dict = dict()
    #iterate over location
    for i in range(0,len(Location),1):
        df_All_Data_dict[Location[i]]= dict()
        Sensors = list(df_AirU_sensor_dict[Location[i]])
        #get sensor and Separate  by sensor
        for j in range(0,len(Sensors),1):
    
            date_list_sensor = df_AirU_sensor_dict[Location[i]][Sensors[j]]['date'].tolist()
            Start_Time_sensor = date_list_sensor[0]
            End_Time_sensor = date_list_sensor[-1]
            
            date_list_DAQ = df_DAQ_values_dict[Location[i]]['Date'].tolist()
            Start_Time_DAQ = date_list_DAQ[0]
            End_Time_DAQ = date_list_DAQ[-1]
            
            #various logics to determine the start and end for Both datasets
            if Start_Time_sensor>Start_Time_DAQ:
                Start_Time_overall = Start_Time_sensor
                
                if End_Time_DAQ>End_Time_sensor:
                    End_Time_overall = End_Time_sensor
                    
                else:
                    End_Time_overall = End_Time_DAQ
            else:
                Start_Time_overall = Start_Time_DAQ
    
                if End_Time_DAQ>End_Time_sensor:
                    End_Time_overall = End_Time_sensor
                    
                else:
                    End_Time_overall = End_Time_DAQ
            
            Air_U_Delete = []            
            DAQ_Delete = []            
            #iterate over entire datset that checks data that is before the start
            #and after the end for AirU
            for k in range(0,len(date_list_sensor),1):
                
                if Start_Time_overall>date_list_sensor[k]:
                    Air_U_Delete.append(k)
                    
                elif End_Time_overall<date_list_sensor[k]:
                    Air_U_Delete.append(k)
                    
            if len(Air_U_Delete)>0:
                
                df_AirU_sensor_dict[Location[i]][Sensors[j]] = \
                df_AirU_sensor_dict[Location[i]][Sensors[j]].drop(Air_U_Delete)
                
                df_AirU_sensor_dict[Location[i]][Sensors[j]] = \
                df_AirU_sensor_dict[Location[i]][Sensors[j]].reset_index()
                
                del df_AirU_sensor_dict[Location[i]][Sensors[j]]['index']

            #Do this for DAQ 
            for k in range(0,len(date_list_DAQ),1):
                if Start_Time_overall>date_list_DAQ[k]:
                    DAQ_Delete.append(k)
                elif End_Time_overall<date_list_DAQ[k]:
                    DAQ_Delete.append(k)
            #Delete all the data that is not needed
            if len(DAQ_Delete)>0:
                
                df_DAQ_values_dict[Location[i]] =\
                df_DAQ_values_dict[Location[i]].drop(DAQ_Delete)
                
                df_DAQ_values_dict[Location[i]] =\
                df_DAQ_values_dict[Location[i]].reset_index()
                
                del df_DAQ_values_dict[Location[i]]['index']

            df_All_Data_dict[Location[i]][Sensors[j]] = pd.concat([df_AirU_sensor_dict[Location[i]][Sensors[j]],\
                             df_DAQ_values_dict[Location[i]]],axis = 1,join = 'outer')
            
            del df_All_Data_dict[Location[i]][Sensors[j]]['Date']
    return df_All_Data_dict
    

"""
Title:
null_code_DAQ_filter

Description:
Removes data that has been nulled by DAQ

Function Dependencies:
Python 3
os
pandas


Inputs:
df_DAQ_Null_Code_dict - dictionaries of dataframe of Null Code dtype = dict 
df_DAQ_values_dict - dictionaries of dataframe of values dtype = dict 

Optional Inputs:
None 

Outputs:
df_DAQ_values_dict - clean dictionary of data frame of values dtype = dict

Optional Outputs:
None 
"""


def null_code_DAQ_filter(df_DAQ_Null_Code_dict,df_DAQ_values_dict):
    #standard setup
    Location_name = list(df_DAQ_Null_Code_dict)
    for i in range(0,len(df_DAQ_Null_Code_dict),1):
        header = list(df_DAQ_Null_Code_dict[Location_name[i]])
        #iterate over the headers
        if header[0]=='Date':
            del header[0]
        for j in range(0,len(header),1):
            #check where there is nullcode
            nullcode = list(df_DAQ_Null_Code_dict[Location_name[i]][header[j]])
            unique_nullcode = list(set(nullcode))
            nan_index = unique_nullcode.index(np.nan)
            #if there is null code put a Nan value that can be cleaned up by 
            #the DAQ cleaner function
            del unique_nullcode[nan_index]
            for k in range(0,len(unique_nullcode)):
                remove_index_list = indexall(nullcode,unique_nullcode[k])
                df_DAQ_values_dict[Location_name[i]][header[j]][remove_index_list] = np.nan
    return df_DAQ_values_dict

"""
Title:
Add_Missing_Data

Description:
Adds data that is missing from the current DAQ dataset-First Location MUST have 
the data and the second/third/...ect can be added later

Function Dependencies:
Python 3
os
pandas


Inputs:
df_DAQ_Null_Code_dict - dictionaries of dataframe of Null Code dtype = dict 
df_DAQ_values_dict - dictionaries of dataframe of values dtype = dict 
missing_var - Variable that needs to be added to a dataset dtype = str

Optional Inputs:
None 

Outputs:
df_DAQ_values_dict - dictionary of data frame of values with missing data dtype = dict
df_DAQ_Null_Code_dict - dictionary of data frame of Null Code with missing data dtype = dict

Optional Outputs:
None 
"""


def Add_Missing_Data(df_DAQ_values_dict,df_DAQ_Null_Code_dict,missing_var = 'SR Value'):
    #get location name
    Location_name = list(df_DAQ_values_dict)
    for i in range(0,len(Location_name),1):
        #check if missing variable is in or not if it is copy and add it to the end
        if missing_var in list(df_DAQ_values_dict[Location_name[i]]):
            Temp_Store = deepcopy(df_DAQ_values_dict[Location_name[i]][missing_var])
            del df_DAQ_values_dict[Location_name[i]][missing_var]
            df_DAQ_values_dict[Location_name[i]] = \
             pd.concat([df_DAQ_values_dict[Location_name[i]],Temp_Store],axis = 1,join = 'outer')

            Temp_Store_NC = deepcopy(df_DAQ_Null_Code_dict[Location_name[i]][missing_var])
            del df_DAQ_Null_Code_dict[Location_name[i]][missing_var]
            df_DAQ_Null_Code_dict[Location_name[i]] = \
             pd.concat([df_DAQ_Null_Code_dict[Location_name[i]],Temp_Store_NC],axis = 1,join = 'outer')
        #if it is not then add it from the stored one 
        else:
            df_DAQ_values_dict[Location_name[i]] =\
            pd.concat([df_DAQ_values_dict[Location_name[i]],Temp_Store],axis = 1,join = 'outer')

            df_DAQ_Null_Code_dict[Location_name[i]] = \
             pd.concat([df_DAQ_Null_Code_dict[Location_name[i]],Temp_Store_NC],axis = 1,join = 'outer')

    return df_DAQ_values_dict,df_DAQ_Null_Code_dict

"""
Title:
Organize_Clean_All

Description:
Used to delete/replace missing values if 2 NaN values are next to each other
the entire row of data will be deleted for all data

Function Dependencies:
Python 3
os
pandas

Inputs:
df_All_Data_dict - dictionary of combined AirU/DAQ dtype = dict

Optional Inputs:
None 

Outputs:
df_All_Data_dict - outputs a clean dictionary of all data values dtype = dict

Optional Outputs:
None 
"""


def Organize_Clean_All(df_All_Data_dict):
    Location_name = list(df_All_Data_dict)

    for i in range(0,len(Location_name),1):
        #check based on sensor
        Sensor_name = list(df_All_Data_dict[Location_name[i]])
        for j in range(0,len(Sensor_name)):
            #see organize_clean_daq
            logic_NAN = np.array(df_All_Data_dict[Location_name[i]][Sensor_name[j]].isnull())
            column_header = list(df_All_Data_dict[Location_name[i]][Sensor_name[j]])
            logic_nan_shape = np.shape(logic_NAN)
            #columns_null = np.unique(test2[1],return_counts=True)
            delete_index_list = []
            #logic test iterate on each row
            for k in range(0,logic_nan_shape[0],1):
                sum_row = np.sum(logic_NAN[k,:])
                if sum_row>0:
                    up_down_check = np.sum((logic_NAN[k,:]*logic_NAN[k-1,:])+(logic_NAN[k,:]*logic_NAN[k+1,:]))
                    if up_down_check>0:
                        delete_index_list.append(k)
                    else:
                        patch_loc = np.where(logic_NAN[k,:])[0]
                        #replace if a nan value if there is a single nan value
                        for l in range(0,len(patch_loc),1):
                            m = patch_loc[l]
                            replace_value = (df_All_Data_dict[Location_name[i]][Sensor_name[j]][column_header[m]][k+1]+\
                             df_All_Data_dict[Location_name[i]][Sensor_name[j]][column_header[m]][k-1])/2
                                             
                            df_All_Data_dict[Location_name[i]][Sensor_name[j]].at[k,column_header[m]] = replace_value
            df_All_Data_dict[Location_name[i]][Sensor_name[j]] = df_All_Data_dict[Location_name[i]][Sensor_name[j]].drop(delete_index_list)
            df_All_Data_dict[Location_name[i]][Sensor_name[j]] = df_All_Data_dict[Location_name[i]][Sensor_name[j]].reset_index()
            del df_All_Data_dict[Location_name[i]][Sensor_name[j]]['index']
    return df_All_Data_dict

"""
Title:
Reorder_df

Description:
Reorder dataframe to make sure that the columns allways match up
    
Function Dependencies:
Python 3
os
pandas

Inputs:
df_All_Data_dict_clean - clean dictionary of dataframe dtype = dict

Optional Inputs:
None 

Outputs:
df_reorder - outputs a reordered version of df_All_Data_dict_clean dtype = dict

Optional Outputs:
None 
"""


def Reorder_df(df_All_Data_dict_clean):
    #copy the two dictionaries
    Location_name = list(df_All_Data_dict_clean)
    df_reorder = dict()
    for i in range(0,len(Location_name),1):
        #get sensor list
        df_reorder[Location_name[i]] = dict()
        sensor_list = list(df_All_Data_dict_clean[Location_name[i]])
        for j in range(0,len(sensor_list),1):
            #reorder the headers
            headers = list(df_All_Data_dict_clean[Location_name[i]][sensor_list[j]])
            headers.sort(key=str.lower)
            df_reorder[Location_name[i]][sensor_list[j]] =\
            df_All_Data_dict_clean[Location_name[i]][sensor_list[j]][headers]
    return df_reorder

"""
Title:
Reorder_df

Description:
Export CSVs given the dictionary dataframe and the path
    
Function Dependencies:
Python 3
os
pandas

Inputs:
df_All_Data_dict_clean - Dictionary data frame that is to be exported dtype = dict
Export_path - export csvs to file

Optional Inputs:
None 

Outputs:
None

Optional Outputs:
None 
"""


def Export_CSV_ALL(df_All_Data_dict_clean,Export_path):
    #get location
    location = list(df_All_Data_dict_clean)
    for i in range(0,len(location),1):
        sensor = list(df_All_Data_dict_clean[location[i]])
        #get sensor
        for j in range(0,len(sensor),1):
            #export csv
            name = location[i]+'_'+sensor[j]+'.csv'
            fullexportpath = os.path.join(Export_path,name)
            df_All_Data_dict_clean[location[i]][sensor[j]].to_csv(fullexportpath,index=False)
    print('Export Complete')

"""
Title:
Remove_AirU_timezone

Description:
Remove AirU timezones
    
Function Dependencies:
Python 3
pandas

Inputs:
df_AirU_sensor_dict - AirU sensor dictionary dtype = dict

Optional Inputs:
None 

Outputs:
df_AirU_sensor_dict - new cleaned AirU sensor dictionary dtype = dict


Optional Outputs:
None 
"""


def Remove_AirU_timezone(df_AirU_sensor_dict):
    location_name = list(df_AirU_sensor_dict)
    for i in range(0,len(location_name),1):
        sensor_name = list(df_AirU_sensor_dict[location_name[i]])
        for j in range(0,len(sensor_name),1):
            temp_date = df_AirU_sensor_dict[location_name[i]][sensor_name[j]]['date'].copy()
            df_AirU_sensor_dict[location_name[i]][sensor_name[j]]['date'] =\
            temp_date.dt.tz_localize(None)
    return df_AirU_sensor_dict
