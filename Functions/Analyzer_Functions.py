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
from matplotlib import rcParams
import numpy as np
import datetime 
from copy import deepcopy as dc

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
fig - matplotlib figure dtype = matplotlib

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
                  filetype='.pdf',dpi_set = 300,padset = 0.1):
    #set date
    today_date = datetime.datetime.today()
    date_report = (str(today_date.strftime('%d-%m-%Y')))
    official_name = plot_name+date_report+filetype
    #get path
    fname = os.path.join(export_path,official_name)
    #save plot
    plt.tight_layout(pad = padset)
    fig.savefig(fname,dpi = dpi_set)
"""
Title : plot_settings

Description - converts plots to publication quality settings

Dependencies 
python 3
matplotlib

Inputs
ax_size - axis size dtype = int
x_tick_size - x label size dtype = int
y_tick_size - y label size dtype = int
figure_size - size of figure dtype = int

Optional Inputs 
None

Outputs
None

Optional Outputs
None
"""


def plot_settings(ax_size,x_tick_size,y_tick_size,figure_size):
    rcParams['axes.labelsize'] = ax_size
    rcParams['xtick.labelsize'] = x_tick_size
    rcParams['ytick.labelsize'] = y_tick_size
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
#    rcParams['text.usetex'] = True
    rcParams['figure.figsize'] = figure_size


"""
Title : plot_parity

Dependencies 
python 3
matplotlib

Description: Plots parity plots given a validated and predicted values

Inputs
yvalid - Actual data dtype = numpy array
ypred - Predicted data dtype = numpy array
xlabel - label x axis dtype = str
ylabel - label y axis dtype = str

Optional Inputs:
c - color dtype = str
s - size of dots dtype = float

Outputs
fig - matplotlib figure dtype matplotlib object

Optional Outputs:
None    
    
"""
def plot_parity(yvalid,ypred,xlabel,ylabel,c = 'b', s = 1):
    fig = plt.figure()
    xy = np.linspace(0,1,100)
    plt.plot(xy,xy,'k')
    plt.scatter(yvalid,ypred,c= c,s= s)
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    return fig


"""
Title : Parameter_Analysis

Dependencies 
python 3
matplotlib
keras 
TensorFlow

Description: Analyzes what each parameter does in the neural network

Inputs
model - Neural Network dtype = tensorflow object
header - list of parameter names dtype = list
X_header_list - list of parameter indexes dtype = list

Optional Inputs:
bounds - should have bounds between 0 and 1? dtype = bool

Outputs
fig_save- dictionary of figures dtype = dict
plt_name_save - list of names to export dtype = list
range_save - numpy array of the range 

Optional Outputs:
None    
    
"""
def Parameter_Analysis(model,header,X_header_list,bounds=False):
    input_dim_ = len(X_header_list)
    rows = 500
    range_save = np.zeros(input_dim_)
    fig_save = dict()
    plt_name_save = []
    for i in range(0,input_dim_,1):
        #give average value of 0.5
        X_valid = np.ones([rows,input_dim_])*0.5
        #change one parameter
        X_valid[:,i] = np.linspace(0,1,rows)
        Y_pred = model.predict(X_valid)
        #zero so that it its easier to read
        Y_zero = Y_pred-np.min(Y_pred)
        #plot figure
        fig_save[i] = plt.figure()
        plt.scatter( X_valid[:,i],Y_zero,c = 'r',s=10)    
        plt.xlabel('Input Value')
        plt.ylabel('Output Value')
        if bounds:
            plt.xlim(-0.1,1.1)
            plt.ylim(-0.1,1.1)
        plt.tight_layout()
        plt_name_save.append(header[X_header_list[i]]+'_')
        range_save[i] = np.max(Y_pred)-np.min(Y_pred)
    return fig_save,plt_name_save,range_save

