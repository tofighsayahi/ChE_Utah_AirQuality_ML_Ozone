# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:20:23 2019

@author: Tim
"""
import_script_path = r"D:\Python_Repository\che_utah_air_quality_group\Functions"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
os.chdir(import_script_path)
from Analyzer_Functions import export_graphs


def extract_data(full_data_path):
    open_data = open(full_data_path,'r')
    data_list = open_data.read().split('\n')
    open_data.close()
    delete_list = []
    for i in range(0,len(data_list),1):
        if data_list[i]=="-------------":
            delete_list.append(i)
        elif data_list[i]=="":
            delete_list.append(i)

        else:
            data_list[i]=data_list[i].replace(" ","")
    if len(delete_list)>0:
        for j in range(0,len(delete_list),1):
            del data_list[delete_list[::-1][j]]
    
    len_data_list = len(data_list)
    name = []
    data = []
    for i in range(0,len_data_list,1):
        name_data = data_list[i].split('=')
        name.append(name_data[0])
        data.append(name_data[1])
    name_cat = list(set(name))
    column = len(name_cat)
    row = int(len(data)/column)
#    print(column)
#    print(row)
    data_array = np.zeros([row,column])
    for i in range(0,row,1):
        for j in range(0,column,1):
            data_array[i,j] = float(data[(i)*(column)+j])
    return data_array,name_cat


plt.close('all')
full_data_path = []

full_data_path.append(r"D:\AirQuality _Research\Data\Nodes_Layers\hyperopt_10.txt")

for i in range(0,len(full_data_path),1):
    data_array,name_list = extract_data(full_data_path[i])
    name_list = ['Error','Nodes','Layers']
    x = data_array[:,1:]
    y = data_array[:,0]
    shape = np.shape(x)
    
    size = ((y/np.max(y)))*100
    fige = plt.figure()
    plt.scatter(x[:,1], x[:,0], c=y,s=size, cmap=cm.viridis)
    plt.xlabel('Layers')
    plt.ylabel('Nodes')
    plt.colorbar()
    name = 'layer_node'
    export_graphs(name,fig=fige,filetype = '.jpg')


    
    for j in range(0,shape[1],1):
        fige = plt.figure()

        temp_max = np.max(x[:,j])
#        x_norm = x[:,j]/temp_max
#        y_max = np.max(y)
#        y_norm = y/y_max
        plt.scatter(x[:,j],y,s = 20)
        plt.xlabel(name_list[j+1])
        plt.ylabel('MSE Loss')
        plt.title('Hyperparameterization Results: '+ name_list[j+1])
        name = 'Hyperparameterization_'+ name_list[j+1]
        export_graphs(name,fig=fige,filetype = '.jpg')

#        plt.legend()
    best_error = np.min(np.min(y))
    optimal = np.where(y == best_error)[0][0]
    print('Report')
    print('------')
    for j in range(0,len(name_list),1):
        print(name_list[j]+' : %0.3f' %data_array[optimal,j])
    
    
    nodes = x[:,0]
    layers = x[:,1]
    

    min_nodes = 0
    max_nodes = 300
    min_layer = 0
    max_layer = 20
    dN = 25
    dL = 5
    nodes_array = np.arange(min_nodes,max_nodes+1e-6,dN,dtype = int)
    layer_array = np.arange(min_layer,max_layer+1e-6,dL,dtype = int)
    Error_array = np.zeros([len(layer_array)-1,len(nodes_array)-1])
    Count_array = np.zeros([len(layer_array)-1,len(nodes_array)-1])
    for j in range(0,len(nodes_array)-1,1):
        for k in range(0,len(layer_array)-1,1):
            Loc = np.where((nodes >= nodes_array[j]) &\
                                 (nodes < nodes_array[j+1]) &\
                                  (layers>=layer_array[k]) &\
                                  (layers<layer_array[k+1]))[0]
            Error_array[k,j] = np.average(y[Loc])
            Count_array[k,j] = len(Loc)
    fige = plt.figure(figsize=(20,10))
    
    plt.imshow(Error_array,cmap = 'viridis')
    plt.xticks(range(len(nodes_array)-1), nodes_array[:-1])
    plt.yticks(range(len(layer_array)-1), layer_array[:-1])

    plt.colorbar()
    name = 'boxspot'
    plt.xlabel('Nodes')
    plt.ylabel('Layers')
    plt.tight_layout()
    export_graphs(name,fig=fige,filetype = '.jpg')
    
    fige = plt.figure(figsize=(20,10))
    plt.imshow(Count_array,cmap = 'viridis')
    plt.xticks(range(len(nodes_array)-1), nodes_array[:-1])
    plt.yticks(range(len(layer_array)-1), layer_array[:-1])
    plt.xlabel('Nodes')
    plt.ylabel('Layers')

    plt.colorbar()
    name = 'boxspot1'
    plt.tight_layout()

    export_graphs(name,fig=fige,filetype = '.jpg')

