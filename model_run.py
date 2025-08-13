#!/usr/bin/env python
# coding: utf-8



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
import random
from sklearn.utils.class_weight import compute_class_weight

import pytorch_lightning as pl
import copy
import logging

from source_code import *
from module_train_utility import *


torch.cuda.is_available()



seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




from test_load import *




'''
Creading Models
'''
#device
device=(
    'cuda'
    if torch.cuda.is_available()
    #else 'mps'
    #if torch.mps.is_available()
    else 'cpu'
)
print(f'using {device} device')




img_dir = 'your/img_dir/'
names=[file for file in os.listdir(img_dir) if file.endswith('.png')]
names = [os.path.splitext(name)[0] for name in names]

from PatchClassifier import *


#define model
model = PatchClassifier(fig_size=128,
                       dropout=0.2,
                       n_pos=135,
                       kernel_size=4,
                       patch_size=8,
                       num_class=4,
                       depth1=2,
                       depth2=8,
                       depth3=4,
                       heads=16,
                       channel=32,
                       policy='mean')



os.chdir('your/work_dir/')
os.getcwd()

model.load_state_dict(torch.load('train_model_weights.pth')) 
model=model.to(device)


for name in names:
    mylist = []
    mylist.append(name)

    test_dataset = test_load(adj=True,crops=128,neighs=8,prune='Grid',names=mylist)
    meta_filter = test_dataset.meta_filter_dict[name]
    test_dataloader = DataLoader(test_dataset,batch_size=1,num_workers=2,shuffle=True)

    model.eval()

    with torch.no_grad():
        for batch, (patches,position,_,adj) in enumerate(test_dataloader):
            patches,position,adj = patches.to(device),position.to(device),adj.to(device)
            
            preds=model(patches,position,adj)
            _,probs=torch.max(preds,1)
            
            position = pd.DataFrame(position.squeeze(0).cpu())
            position=position.rename(columns={0:'coord_x',1:'coord_y'})
            position['TLS_score'] = probs.cpu().numpy()
            print(f'name: {name}; score: {set(probs.cpu().numpy())}')

            os.mkdir(f'your/save_dir/{name}')
            position.to_csv(f'your/save_dir/{name}/position.csv')
    print(name)
        
    torch.cuda.empty_cache()
    

