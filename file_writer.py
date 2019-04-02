#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:16:48 2018

@author: jnickell
"""

import os
import pandas as pd
import numpy as np

def df_2_txt_confusion_matrix(df, filename=os.getcwd()+'/', write_type='a', dtype='d'):
    
# pull data stored within DataFrame    
    
    col_names = np.array(df.columns, dtype='str')
    row_names = np.array(df.index, dtype='str')
    row_names = np.concatenate((['True \ Predicted'], row_names))
    str_vals = df.values.astype(str)
    spaces = len(max(col_names))+4 
    len_format = '{:>'+str(spaces)+'s}'
    
    for el in range(str_vals.shape[0]):
        str_vals[el] = ([len_format.format(x) for x in str_vals[el]])
    
    col_names = np.core.defchararray.rjust(col_names, spaces)
    row_names = np.core.defchararray.rjust(row_names, spaces)
    
    new_array = np.vstack((col_names, str_vals))

    row_names = np.transpose([row_names])
    final_array = np.concatenate((row_names, new_array), axis=1)
    
    with open(filename, write_type) as file:
        for item in final_array:
            file.writelines(list(item))
            file.write('\n')
    
    
    