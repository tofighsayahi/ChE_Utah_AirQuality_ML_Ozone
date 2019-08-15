# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:32:25 2019

@author: Tim
"""

import numpy 
import pandas 
import keras 
import tensorflow
import sklearn
import matplotlib
import scipy
package_list = [numpy,pandas,keras,tensorflow,sklearn,matplotlib,scipy]
package_name = ['numpy','pandas','keras','tensorflow','sklearn','matplotlib','scipy']

for i in range(0,len(package_name),1):
    print(package_name[i]+' : '+str(package_list[i].__version__))