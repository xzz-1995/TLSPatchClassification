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
import torchvision.models as models
from torchmetrics.classification import Accuracy

from tqdm import tqdm
from tempfile import TemporaryDirectory

import skimage.filters as sk_filters
from scipy.spatial import distance_matrix, minkowski_distance, distance
from sklearn.cluster import KMeans
import random
from sklearn.utils.class_weight import compute_class_weight

import pytorch_lightning as pl
import copy
import logging

from module_train_utility import *


class PatchClassifier(pl.LightningModule):
    def __init__(self,
                 fig_size=128,
                 dropout=0.2,
                 n_pos = 135, #maxium of all meta (need adjust)
                 kernel_size=4, #for convmixer block Conv2d
                 patch_size=8, #for patch embedding adjust [n,32,16,16]
                 num_class=4,
                 depth1=2, #convmixer block layer, for adjust
                 depth2=8, #attention block layer, for adjust
                 depth3=4, #gnn block layer, for adjust
                 heads=16, #number of heads in multihead attention in attn block for adjust,
                 channel=32,
                 policy='mean'):
        super().__init__()

        #self parameters
        dim=(fig_size//patch_size)**2*channel//8
        dim_head=dim/heads
    
        #self functions
        ## embedding multi-dimension data to 2D, for ConvDepwise and Convpointwise
        self.patch_embedding=torch.nn.Conv2d(3,channel,patch_size,patch_size)

        self.dropout=nn.Dropout(dropout)

        ## convmixer moduel
        self.layer1=nn.Sequential(
            *[convmixer_block(channel,kernel_size) for i in range(depth1)],
        )

        ## MLP
        self.down=nn.Sequential(
            nn.Conv2d(channel,channel//8,1,1),
            nn.Flatten(),
        )

        ## Add centers
        self.x_embed=nn.Embedding(n_pos,dim)
        self.y_embed=nn.Embedding(n_pos,dim)


        ## Transformer module
        self.layer2 = nn.Sequential(*[attn_block(dim,heads,dim_head,dim,dropout) for i in range(depth2)])
        
        ## GNN module
        self.layer3 = nn.ModuleList([gs_block(dim,dim,policy,True) for i in range(depth3)])


        ## LSTM
        self.jknet = nn.Sequential(
            nn.LSTM(dim,dim,2), #2 means the layer of LSTM, for adjusted
            SelectItem(0), #output of LSTM contains 2 elements, (0) is ct, (1) is hidden layer
        )

        ## MLP to class
        self.class_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_class),
        )


    def forward(self,patches,positions,adj):
        B,N,C,H,W=patches.shape #B, N, 3, 128, 128
        patches=patches.reshape(B*N,C,H,W)
        
        #patch embedding:extracting morphology info from each patch
        patch_emb=self.patch_embedding(patches)  #from [n,3,128,128] to [n,32,16,16]

        #convmixer module
        x=self.dropout(patch_emb)
        x=self.layer1(x)

        #MLP
        x = self.down(x)
        g = x.unsqueeze(0)

        # add centers
        centers_x = self.x_embed(positions[:,:,0])
        centers_y = self.y_embed(positions[:,:,1])
        ct = centers_x+centers_y

        #transformer module
        layer2_input = g+ct
        g=self.layer2(layer2_input).squeeze(0)

        #gnn module
        jk = []
        for layer in self.layer3:
            g = layer(g,adj)
            jk.append(g.unsqueeze(0))
        
        g = torch.cat(jk,0)

        #LSTM
        g = self.jknet(g).mean(0)

        #MLP to num_class
        x = self.class_head(g)
        
        
        #prob of each class
        #prob = F.softmax(pred,dim=1)
        
        return x
