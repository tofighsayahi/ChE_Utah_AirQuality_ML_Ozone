# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:20:23 2019

@author: Tim
"""

def extract_data(full_data_path,var):
    open_data = open(full_data_path,'r')
    data_list = open_data.read().split('|')
    open_data.close()
    rowcount=0
    delete_list = []
    for j in range(0,len(data_list),1):
        if data_list[j]=="":
            delete_list.append(j)
        elif data_list[j]==" ":
            delete_list.append(j)
        elif  data_list[j]=='\n':
            rowcount+=1
            delete_list.append(j)

        else:
            data_list[j]=data_list[j].replace(" ","")
    if len(delete_list)>0:
        for j in range(0,len(delete_list),1):
            del data_list[delete_list[::-1][j]]

    name_list = data_list[0:var]
    data_array = np.zeros([rowcount,var])
    for j in range(0,rowcount,1):
        for k in range(0,var,1):
            data_array[j,k] = float(data_list[(j+1)*(var)+k])
    return data_array,name_list

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.close('all')
full_data_path = []

full_data_path.append(r"D:\AirQuality _Research\Data\Nodes_Layers\hyperopt_1.txt")
full_data_path.append(r"D:\AirQuality _Research\Data\Nodes_Layers\hyperopt_2.txt")
full_data_path.append(r"D:\AirQuality _Research\Data\Nodes_Layers\hyperopt_3.txt")

#data_list = []
#name_list = []
#data_array_list = []
var = np.array([8,8,4])
for i in range(0,len(full_data_path),1):
    data_array,name_list = extract_data(full_data_path[i],var[i])
    x = data_array[:,2:]
    y = data_array[:,1]
    shape = np.shape(x)
    if i==2:
        plt.figure()
        plt.scatter(x[:,1], x[:,0], c=-y, cmap=cm.inferno)
        plt.xlabel('Nodes')
        plt.ylabel('Layers')
        plt.colorbar()


    
    else:
        for j in range(0,shape[1],1):
            plt.figure()
    
            temp_max = np.max(x[:,j])
            x_norm = x[:,j]/temp_max
            y_max = np.max(-y)
            y_norm = -y/y_max
            plt.scatter(x_norm,y_norm,s = 20)
            plt.xlabel('Parameter')
            plt.ylabel('MSE Loss')
            plt.title('Hyperparameterization Results:'+ name_list[2+j])
    #        plt.legend()
        best_error = np.min(np.min(-y))
        optimal = np.where(y == -best_error)[0][0]
        print('Report')
        print('------')
        for j in range(0,len(name_list)-1,1):
            print(name_list[j+1]+' : %0.3f' %data_array[optimal,j+1])
            
            