#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:10:48 2019

@author: aj611
"""

from PIL import Image
import numpy as np

for i in range(1, 6):
    file_name = "data_batch_%d"%(i)
    file = open(file_name, 'ab')
    
    for j in range(1, 1000):
        num = (i - 1) * 1000 + j
        image_name = "%d.jpeg"%(num)
        
        
        im = Image.open(image_name)
        im = (np.array(im))
    
        r = im[:, :, 0].flatten()
        g = im[:, :, 1].flatten()
        b = im[:, :, 2].flatten()
        
        label = [11]
        
        out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
        out.tofile(file)