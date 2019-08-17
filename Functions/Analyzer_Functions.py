# -*- coding: utf-8 -*-
"""
Title : Analyzer Function

Description : This the functions that help make it a bit easier to analyze and export 
results generated from neural networks

Function Dependencies:
keras
numpy
random
matplotlib
TensorFlow
os 
datetime
python
    
Created on Tue Jun 16 21:36:12 2019
Revised and Commented 08/15/2019
@author: Timothy Quah
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import datetime 


"""
Title:
export_graphs

Description: Used to export figure
    
Function Dependencies:
Python 3
datetime
matplotlib
os

Inputs:
plot_name - name of plot dtype = str
fig - matplotlib figure dtype = matpl

Optional Inputs:
export_path - path to export to dtype = str
filetype- export file type dtype = str
dpi_set - resolution dtype = int
    
Outputs:
Nonels

Optional Outputs:
None 
"""

def export_graphs(plot_name,fig,export_path = r"D:\AirQuality _Research\Plots",\
                  filetype='.pdf',dpi_set = 300):
    #set date
    today_date = datetime.datetime.today()
    date_report = (str(today_date.strftime('%d-%m-%Y')))
    official_name = plot_name+date_report+filetype
    #get path
    fname = os.path.join(export_path,official_name)
    #save plot
    fig.savefig(fname,dpi = dpi_set)



