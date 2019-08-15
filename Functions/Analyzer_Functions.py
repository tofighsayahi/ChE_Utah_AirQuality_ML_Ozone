# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:20:07 2019

@author: Tim
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def export_graphs(plot_name,fig,export_path = r"D:\AirQuality _Research\Plots",\
                  filetype='.pdf',dpi_set = 300):
    import datetime 
    today_date = datetime.datetime.today()
    date_report = (str(today_date.strftime('%d-%m-%Y')))
    official_name = plot_name+date_report+filetype
    fname = os.path.join(export_path,official_name)
    fig.savefig(fname,dpi = dpi_set)



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

