import os
import pandas as pd
import numpy as np
import cv2 as cv
import math 
import gmpy as g
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as tf

from tqdm import tqdm

import skimage.filters as sk_filters
from scipy.spatial import distance_matrix, minkowski_distance, distance
import random

import copy

from source_code import *

class test_load(torch.utils.data.Dataset):
    def __init__(self,
                 crops= 128,
                 adj=True,
                 prune='Grid',
                 neighs=8,
                 names = ['63779', '12212']):
        super(test_load,self).__init__()

        self.img_dir = 'your_image_dir/'
        #self.meta_dir = 'anno_meta/'
        
        self.names = names

        self.adj = adj

        #print('Loading imgs...')
        self.img_dict= {i:get_img(self.img_dir,i) for i in self.names}
        
        #print('Load imgs meta...')
        self.meta_dict={
            key:get_meta(value,crops) 
            for key, value in self.img_dict.items()
        }

        #print('Tiling imgs...')
        self.patch_dict={}
        self.den_dict={}
        for key, value in self.img_dict.items(): #change to exclude norm
            meta=self.meta_dict[key]
            img_val,den=get_patch(value,meta,crops)
            self.patch_dict[key]=img_val
            self.den_dict[key] = den

        #print('Filter meta...')
        self.meta_filter_dict={}
        self.meta_dictA={}
        for key,value in self.meta_dict.items():
            den=self.den_dict[key]
            meta,meta_f=meta_filter_fun(value,den)
    
            self.meta_filter_dict[key]=meta_f
            self.meta_dictA[key]=meta
        
        #self.lbl_dict={key:torch.tensor(value['TLS_score'].values-1) for key, value in self.meta_filter_dict.items()}

        #print('Numpy array img to torch...')
        self.patch_tensor_dict={}
        for key,value in self.patch_dict.items():
            meta_filter=self.meta_filter_dict[key]
            patch_tensor=np_to_tensor(meta_filter,value)
            self.patch_tensor_dict[key]=patch_tensor

        #print('Get img coord and centers...')
        #patch coordinates
        self.coord_dict = {}
        for key, value in self.meta_filter_dict.items():
            coord = value[['coord_x','coord_y']].values.astype(np.float32)
            #coord_tensor = torch.from_numpy(coord)
            self.coord_dict[key]=coord

        #pixel/image centers
        self.centers_dict={}
        for key, value in self.meta_filter_dict.items():
            centers = value[['image_x','image_y']].values.astype(np.float32)
            self.centers_dict[key]=centers

        #print('Calculate img adjacent patches use X/Y coordinates...')
        self.adj_dict = {i:calcADJ(m,neighs,pruneTag=prune) for i,m in self.coord_dict.items()}

        ## filter with adj
        ### label of patches without neighbors
        self.adj_f_dict = {}
        self.patches_f_dict = {}
        self.coord_f_dict = {}
        self.centers_f_dict = {}
        #self.lbl_f_dict = {}
        
        for key,value in self.meta_filter_dict.items():
            adj = self.adj_dict[key]
            num_neigh = adj.sum(1,keepdim = True)
            label = num_neigh.nonzero()
            label = label[:,0]

            ### for adj
            adj_f = adj[:,label]
            adj_f = adj_f[label,:]
            self.adj_f_dict[key]=adj_f

            ### for patches
            patches = self.patch_tensor_dict[key]
            patches_f = patches[label,:,:,:]
            self.patches_f_dict[key]=patches_f

            ###for coord
            coord = self.coord_dict[key]
            coord_f = coord[label,:]
            self.coord_f_dict[key]=coord_f
        
            ### for centers
            centers = self.centers_dict[key]
            centers_f = centers[label,]
            self.centers_f_dict[key]=centers_f

            '''### for lbls
            lbls = self.lbl_dict[key]
            lbls_f = lbls[label]
            self.lbl_f_dict[key]=lbls_f
            '''

        self.id2name = dict(enumerate(self.names))
    
    def __getitem__(self,index):
        ID=self.id2name[index]
        patches_f=self.patches_f_dict[ID]
        coord_f=self.coord_f_dict[ID]
        positions = torch.LongTensor(coord_f)
        centers_f=self.centers_f_dict[ID]
        centers_ft = torch.LongTensor(centers_f)
        adj_f=self.adj_f_dict[ID]
        #label_f=self.lbl_f_dict[ID]
        
        data = [patches_f,positions,centers_ft]#,label_f]
        if self.adj:
            data.append(adj_f)

        return data
    def __len__(self):
        return len(self.centers_f_dict)
