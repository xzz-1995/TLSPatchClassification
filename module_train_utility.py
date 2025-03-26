import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import pandas as pd

from torch import nn, einsum
from scipy.stats import pearsonr
from torch.autograd import Function
from torch.autograd.variable import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class convmixer_block(nn.Module):
    def __init__(self,dim,kernel_size):
        super().__init__()
        #convDepwise
        self.dw=nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.BatchNorm2d(dim),
                nn.GELU(),
        )
        #convPointwise
        self.pw=nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
    def forward(self,x):
        #el = BN(s{convDepwise(el-1)})+el-1
        x=self.dw(x)+x 
        #el+1 = BN(s{convPointwise(el)})
        x=self.pw(x)
        return x


class attn_block(nn.Module):
    def __init__(self,dim,heads,dim_head,mlp_dim,dropout = 0.):
        super().__init__()
        self.attn = PreNorm(dim,Attention(dim,heads,dim_head,dropout))
        self.ff = PreNorm(dim,FeedForward(dim,mlp_dim,dropout))
    def forward(self,x):
        x = self.attn(x)+x
        x = self.ff(x)+x
        return x


class PreNorm(nn.Module):
    def __init__(self,dim,fn): # fn represents a function, in this case is the following Attention func
        super().__init__()
        self.norm = nn.LayerNorm(dim) # normalization 1024 dimension
        self.fn = fn
    def forward(self,x,**kwargs): # kwargs: parameters in fn
        return self.fn(self.norm(x), **kwargs) #first norm, then fn

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)
        

class Attention(nn.Module):
    def __init__(self,dim = 1024, heads = 16, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = int(dim_head * heads) # 64*16=1024
        project_out = not(heads==1 and dim_head == dim)

        self.heads = heads              # 16
        self.scale = dim_head ** -0.5   # 0.125

        # create a function called attend, its core function is Softmax, when dim=-1 the same as dim=2, sum a matrix in row
        self.attend = nn.Softmax(dim=-1)
        # convert MLP result 1024 features to q,k,v (1024 * 3)for multihead
        self.to_qkv=nn.Linear(dim,inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() # create a net doing nothing used as the input layer for nn

    def forward(self,x):
        b,n,_,h = *x.shape,self.heads # b = batch_size, n = n_patches, _=1024, h=heads

        
        qkv = self.to_qkv(x).chunk(3,dim=-1) # seperate 1024D features to q,k,v[1,npatches,3072],chunk(dim=-1) means list[3*[1,npatches,3072/3]]
        q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d',h = h),qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q,k)*self.scale
        # softmax
        attn = self.attend(dots)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class gs_block(nn.Module):
    def __init__(
        self,feature_dim,embed_dim,policy='mean',gcn=True
    ):
        super().__init__()
        self.gcn = gcn
        self.policy=policy
        self.embed_dim = embed_dim #1024
        self.feat_dim = feature_dim #1024

        # create a trianable tensor, passing from layers
        self.weight = nn.Parameter(torch.FloatTensor(
            self.embed_dim,
            self.feat_dim if self.gcn else 2 * self.feat_dim
        ))
        # create an initial weight matrix, with same sd when passing layers, suitable for forward and back ward
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self,x,Adj):
        neigh_feats = self.aggregate(x,Adj)
        if not self.gcn:
            combind = torch.cat([x,neigh_feats],dim=1)
        else:
            combind = neigh_feats

        combind = F.relu(self.weight.mm(combind.T)).T 
        combind = F.normalize(combind,2,1)
        return combind

    def aggregate(self,x,Adj):
        adj = Variable(Adj).to(Adj.device) #conver tensor to variable type value which could be studied

        if not self.gcn:
            n=len(adj)
            adj = adj-torch.eye(n).to(adj.device)

        if self.policy == 'mean':
            num_neigh = adj.sum(1,keepdim = True) #sum in row
            mask = adj.div(num_neigh)             #average of each row
            to_feats = mask.squeeze(0).mm(x)      #mask * layer2 output
        elif self.policy == 'max':
            indexs = [i.nonzero() for i in adj == 1]
            to_feats = []
            for feat in [x[i.squeeze()] for i in indexs]:
                if len(feat.size()) == 1:
                    to_feats.append(feat.view(1,-1))
                else:
                    to_feats.append(torch.max(feat,0)[0].view(1,-1))
            to_feats = torch.cat(to_feats,0)
        return to_feats


class SelectItem(nn.Module):
    def __init__(self,item_index):
        super(SelectItem,self).__init__()
        self.item_index = item_index

    def forward(self,inputs):
        return inputs[self.item_index]


