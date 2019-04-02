#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:10:52 2018

@author: jnickell
"""

import csv
from urllib.request import urlopen
import collections
import numpy as np

def download_online_dat(url, csv_name):
    
    response = urlopen(url)
    read_dat = str(response.read())
    lines = read_dat.split('\\n')
    names = lines[0].replace('"','').split(',')
    names[0] = 'record'
    lines[0]=''
    dat = collections.OrderedDict()
    for name in names:
        lines[0] += name+','
        dat[name] = np.array([])
    lines = lines[1:-1]

    with open(csv_name, 'w') as file:
#        fieldnames=dat.keys()
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        for line in lines:
            this_line = line.replace('"', '').split(',')
            tmp_dict = collections.OrderedDict()
            for i, name in enumerate(names):
                tmp_dict[name] = this_line[i]
            writer.writerow(tmp_dict)
    response.close()