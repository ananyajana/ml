#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:24:39 2019

@author: aj611
"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    
import torch.utils.data as data
#from .utils import check_integrity

class custom_CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    filename = 'cifar-10-python.tar.gz'
    
    train_list = [ #fill in the md5 digests  newly created
        #['data_batch_1','e3806dc0d2b97bf8851497b5736c31cd'],
        #['data_batch_2','fa5a1502784686df90efaa433de050be'],
        #['data_batch_3','a560b2ccff218aadbcd176ba9aa3a194'],
        #['data_batch_4','113d8032f4609dcf3b605d7d183cfe22'],
        #['data_batch_5','831a06f01e0169e2fd7493f382016fc8']
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e']
    ]
    
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '8b08ce5b9915f51e560b77c3aace54ca'# fill in the md5 after modification
    }
        
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
            
        self.data = []
        self.targets = []
        
        # now load the picked numpy array
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding = 'latin1')
                
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                    
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1) #convert to HWC
        
        self._load_meta()
        
    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding = 'latin1')
            
            self.classes = data[self.meta['key']]
        self.class_to_ids = {_class: i for i, _class in enumerate(self.classes)}
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        fmt_str = 'Dataset' + self.__class__.__name__ + '\n'
        fmt_str += ' Number od datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else  'test'
        fmt_str += ' Split: {}\n'.format(tmp)
        fmt_str += ' Root Location: {}\n'.format(self.root)
        tmp = ' Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    

