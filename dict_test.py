#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:42:23 2021

@author: karas
"""
import numpy as np
import operator


# keys = [(i, j) for i in range(3) for j in range(3)]

# values = [i for i in np.random.rand(9)]



# test_dict = dict(zip(keys, values))

test_dict = {(0,0):2.57, (0,1):0.73, (1,0):0.67, (1,1):3.55} 


left_index = [i[0] for i in test_dict.keys()]
right_index = [i[1] for i in test_dict.keys()]


left_index = np.array(left_index)
left_index = np.unique(left_index)

right_index = np.array(right_index)
right_index = np.unique(right_index)



unique = []

last_dict = {}

for l_idx in left_index:
    inter_dict = {}
    for j in list(test_dict.keys()):
        if  l_idx == j[0]:
            inter_dict[j] = test_dict[j]
            
            
    
    x = min(inter_dict.items(), key=operator.itemgetter(1))[0]
    last_dict[x] = test_dict[x]
    
    
    
  



    
