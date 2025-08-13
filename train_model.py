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

import skimage.filters as sk_filters
from scipy.spatial import distance_matrix, minkowski_distance, distance
import random
from sklearn.utils.class_weight import compute_class_weight

import pytorch_lightning as pl
import copy
import logging

from data_load_utility import *
from module_train_utility import *


seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class WSI_load(torch.utils.data.Dataset):
    def __init__(self,
                 train=True,
                 fold=0, #index of test data
                 crops= 128,
                 adj=True,
                 prune='Grid',
                 neighs=8):
        super(WSI_load,self).__init__()

        self.img_dir = 'your/img_dir/'
        self.meta_dir = 'your/anno_meta/'

        names=[file for file in os.listdir(self.meta_dir) if not file.endswith('.DS_Store')]

        self.train=train
        self.adj = adj

        samples=names
        te_names=['Val1','Val2','Val3','Val4','Val5','Val6','Val7','Val8']
        print(te_names)
        tr_names=list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names
        
        print('Loading imgs...')
        self.img_dict= {i:get_img(self.img_dir,i) for i in self.names}

        
        print('Load imgs meta...')
        self.meta_dict={
            key:get_meta(value,crops) 
            for key, value in self.img_dict.items()
        }

        print('Tiling imgs...')
        self.patch_dict={}
        for key, value in self.img_dict.items(): #change to exclude norm
            meta=self.meta_dict[key]
            img_val=get_patch_noden(value,meta,crops)
            self.patch_dict[key]=img_val
            
        print('Load filtered meta...')
        self.meta_filter_dict={i:read_meta(self.meta_dir,i) for i in self.names}
        
        self.lbl_dict={key:torch.tensor(value['TLS_score'].values-1) for key, value in self.meta_filter_dict.items()}

        print('Numpy array img to torch...')
        self.patch_tensor_dict={}
        for key,value in self.patch_dict.items():
            meta=self.meta_dict[key]
            meta_filter=self.meta_filter_dict[key]
            patch_tensor=np_to_tensor(meta,meta_filter,value)
            self.patch_tensor_dict[key]=patch_tensor

        print('Get img coord and centers...')
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

        print('Calculate img adjacent patches use X/Y coordinates...')
        self.adj_dict = {i:calcADJ(m,neighs,pruneTag=prune) for i,m in self.coord_dict.items()}

        ## filter with adj
        ### label of patches without neighbors
        self.adj_f_dict = {}
        self.patches_f_dict = {}
        self.coord_f_dict = {}
        self.centers_f_dict = {}
        self.lbl_f_dict = {}
        
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

            ### for lbls
            lbls = self.lbl_dict[key]
            lbls_f = lbls[label]
            self.lbl_f_dict[key]=lbls_f

        self.id2name = dict(enumerate(self.names))
    
    def __getitem__(self,index):
        ID=self.id2name[index]
        patches_f=self.patches_f_dict[ID]
        coord_f=self.coord_f_dict[ID]
        positions = torch.LongTensor(coord_f)
        centers_f=self.centers_f_dict[ID]
        centers_ft = torch.LongTensor(centers_f)
        adj_f=self.adj_f_dict[ID]
        label_f=self.lbl_f_dict[ID]
        
        data = [patches_f,positions,centers_ft,label_f]
        if self.adj:
            data.append(adj_f)

        return data
    def __len__(self):
        return len(self.centers_f_dict)

os.chdir('your/work_dir/')
os.getcwd()


train_dataset=WSI_load(train=True,fold = 1, adj=True,crops=128,neighs=8,prune='Grid')
train_dataset


datasets = {'train' : train_dataset,
           'val' : test_dataset}
dataloaders = {x:DataLoader(datasets[x],batch_size=1,num_workers=2,shuffle=True) for x in ['train','val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}


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


#define model
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


model = PatchClassifier(fig_size=128,
                       dropout=0.2,
                       n_pos=135,
                       kernel_size=4,
                       patch_size=8,
                       num_class=4,
                       depth1=2,
                       depth2=8,
                       depth3=4,
                       heads=4,
                       channel=32,
                       policy='mean').to(device)


class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 
        :param weight_decay:
        :param p: 2 default, 0 L2, 1 L1.
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#get the newest weight
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        get the newest weight list
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        :param weight_list:
        :param p: 
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")

'''
optimizing the model parameters
'''

#class weight
tr_labs=[]
for key, value in train_dataset.lbl_dict.items():
    tr_labs.append(value)
class_weight = compute_class_weight('balanced',
                                    classes=[0,1,2,3],
                                    y=torch.concat(tr_labs).numpy())

class_weight = torch.Tensor(class_weight)
class_weight = class_weight.to(device)


#weight_decay
weight_decay=10.0 
reg_loss=Regularization(model, weight_decay, p=2).to(device)

loss_fn=nn.CrossEntropyLoss(weight=class_weight).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)#,weight_decay=0.001)
StepLR=torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.9)


def get_logger(filename,verbosity=1,name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter=logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger=logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh=logging.FileHandler(filename,'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh=logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger



def train_model(model,loss_fn,optimizer,StepLR,num_epochs=350):
    
    since=time.time()
    
    for epoch in range(num_epochs):
        
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss=0.0
            running_corrects=0

            for batch, (patches,position,_,label,adj) in enumerate(dataloaders[phase]):
                patches,position,adj,label, = patches.to(device),position.to(device),adj.to(device),label.to(device)
                label = label.squeeze(0)
            
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds=model(patches,position,adj)
                    _,probs=torch.max(preds,1)
                    loss = loss_fn(preds, label)
                    loss = loss + 1e-2 * reg_loss(model)#/patches.squeeze(0).size(0)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            
                running_loss += loss.item() * patches.size(0)
                accuracy = Accuracy(task="multiclass", num_classes=4).to(device)
                running_corrects += accuracy(probs,label)

            if phase == 'train':
                StepLR.step()
            
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            list = [epoch,phase,epoch_loss,format(epoch_acc)]
            data = pd.DataFrame([list])
            data.to_csv('train_model.csv',mode='a',header=False,index = False)
            logger.info('Epoch:[{}/{}]\t {}\t loss={:.4f}\t acc={:.4f}'.format(epoch, num_epochs,phase,epoch_loss,epoch_acc))

        print('Done!')
        time_elapsed=time.time()-since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,time_elapsed % 60))

    return model


logger=get_logger('your_path/train.log')
logger.info('start training...')


df = pd.DataFrame(columns = ['Epoch','Phase','loss','acc'])
df.to_csv('your_path/train_model.csv',index = False)


model=train_model(model,loss_fn,optimizer,StepLR,num_epochs=500)

torch.save(model, 'your_path/train_model.pth')

torch.save(model.state_dict(), 'your_path/train_weight.pth')


