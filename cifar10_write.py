#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:10:48 2019
This file is to append the generated/synthesized  images to the original cifar10 data bins
Input: cifar10 data bin files and the, synthesized images
Output: the cifar10 databin files will have the new images appended to them which can be checked by the increasein size
@author: aj611
"""

from PIL import Image
import numpy as np

for i in range(1, 6):
# modifying the 5 data_batch bin files by appending 1/5 th of the generated
# images to each of them
    file_name = "data_batch_%d"%(i)
    file = open(file_name, 'ab')

# the upperlimit of the range changes according to the number of generated images.
# In this case we had generated 5000 images, of which 200 are getting appened to each individual file    
    for j in range(1, 1000):
# just some mathematica manipulation to make sure the correct generated images get appended to the correct data_batch file
        num = (i - 1) * 1000 + j
        image_name = "%d_IMGP.jpeg"%(num) # we have used imgp tool to resize the generated 64x64 images to 32x32
        
        
        im = Image.open(image_name)
        im = (np.array(im))
    
        r = im[:, :, 0].flatten()
        g = im[:, :, 1].flatten()
        b = im[:, :, 2].flatten()
        
        label = [10] # cifar10 image labels are from 0 to 9. Hence the label of the 11 th class is 10. Does the label number impact, I mean whether 10 or 11?
        
        out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
        out.tofile(file)
    file.close()
