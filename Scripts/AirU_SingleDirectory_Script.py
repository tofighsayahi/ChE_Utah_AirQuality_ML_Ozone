# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:30:12 2019

@author: Tim
"""
## Import programs
import os #used to for operating system functionalities

## Gets program to look in the directory 
function_path = "../Functions" #path to Tim's function

## Change directory to function directory allows script to use functions
os.chdir(function_path)

from Cleaner_Loader_Functions import sub_main_merge

#Where are the AirU directory
import_path_AIR_U =\
r"E:\PhD project\ozone\all data"

#Where do you want to dump the files?
export_path_Air_U = \
r"E:\PhD project\ozone\08212019_All_Data_Single_Directory"
#Function that does this work
sub_main_merge(import_path_AIR_U,export_path_Air_U)
