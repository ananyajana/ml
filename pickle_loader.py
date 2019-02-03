#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:09:36 2019

@author: aj611
"""

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    


meta_dict = unpickle('batches.meta')
print(meta_dict)
print(meta_dict[b"label_names"].append(b'ship_plane'))
print(meta_dict)