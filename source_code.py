#import staintools
import os

#from openslide import open_slide
#import openslide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import cv2 as cv
import gmpy as g
import math 
import pandas as pd

import torch
import torch.nn as nn
from tqdm import tqdm

import seaborn as sns


def get_img(img_dir,names):
        path = img_dir+names+'.png'
        #im = cv.imread(path)
        im = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
        return im


def norm_img(img_dict,names):
    img_dict_norm={}
    std=staintools.LuminosityStandardizer.standardize(img_dict[names[0]])
    
    for i in range(0,len(names)):
        #brightness standardization
        im_stand=staintools.LuminosityStandardizer.standardize(img_dict[names[i]])
    
        #normalize to first image
        normalizer=staintools.StainNormalizer(method='vahadane')
        normalizer.fit(std)
        im_norm=normalizer.transform(im_stand)

        #store the reuslts in the dictionary
        img_dict_norm[names[i]]=im_norm

    return img_dict_norm

def get_meta(img,crops):
    #get tile coord
    col=img.shape[1]
    row=img.shape[0]

    coord_x = g.mpz(math.ceil((col-crops)/(crops-1)))
    coord_y = g.mpz(math.ceil((row-crops)/(crops-1)))

    coord_x = np.arange(0,coord_x+1)
    coord_y = np.arange(0,coord_y+1)

    image_x = (crops-1)*coord_x
    image_y = (crops-1)*coord_y
    
    TT1 = pd.DataFrame(np.array(np.meshgrid(coord_x, coord_y)).T.reshape(-1, 2), columns=['coord_x', 'coord_y'])
    TT2 = pd.DataFrame(np.array(np.meshgrid(image_x, image_y)).T.reshape(-1, 2), columns=['image_x', 'image_y'])

    TT = pd.concat([TT1, TT2], axis=1)
    return(TT)


import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
#tiling image and get the complete gray density
def get_patch(image,meta,crops):
        img_val = []
        den = {}
        with tqdm(
                total=len(meta.index),
                desc="Tiling image",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]",
             ) as pbar:
             for i in range(len(meta.index)):
                #get tiling size
                 
                image_y,image_x = meta.iloc[i][['image_y','image_x']].astype(int)

                image_down = image_y 
                image_up = image_y + crops
                image_left = image_x 
                image_right = image_x + crops

                if image_right > image.shape[1]:
                        image_right = image.shape[1]
                
                if image_up > image.shape[0]:
                        image_up = image.shape[0]

                #tiling from np.array
                tile_hwc = image[int(image_down):int(image_up),int(image_left):int(image_right),:]

                desired_rows = crops
                desired_cols = crops
                
                if tile_hwc.shape[0] < crops or tile_hwc.shape[1] < crops:
                        extended_tile = np.full((desired_rows,desired_cols,tile_hwc.shape[2]),255,dtype=np.uint8)
                        extended_tile[:tile_hwc.shape[0],:tile_hwc.shape[1],:] = tile_hwc
                        tile_hwc = extended_tile

                #to grey scale
                tile_hwc_gs = np.dot(tile_hwc[...,:3],[0.2125, 0.7154, 0.0721])
                tile_hwc_gs = tile_hwc_gs.astype('uint8')
                ###complement greyscale
                tile_hwc_gsc = 255 - tile_hwc_gs

                ###patch mean gray intensity
                tile_den = np.mean(tile_hwc_gsc)
                col,row = meta.iloc[i][['coord_x','coord_y']].astype(int)
                temp_coord = f"{col}_{row}"
                den[temp_coord] = tile_den
                 
                tile_hwc=tile_hwc.astype(np.float32)
                img_val.append(tile_hwc)
                #img_tensor = torch.Tensor(np.array(img_val))
                pbar.update(1)
        
        return img_val, den



import skimage.filters as sk_filters
def meta_filter_fun(meta,den):
        den_df = pd.DataFrame.from_dict(den,orient = 'index')
        den_df.columns = ['density']

        ##remove the rownames of den_df
        den_df.index=[None]*len(den_df)

        ##assign new rownames
        den_df.index=meta.index

        ##combind coord with int_df to get metadata of patches
        meta=pd.concat([meta,den_df],axis=1)

        ##get threshold
        thresh_value=sk_filters.threshold_otsu(np.array(meta['density'].tolist()))
    
        ##filter patches with grey density
        meta['in_tissue']=np.where(meta['density']>thresh_value,1,0)
        meta_filter = meta[meta['in_tissue'] == 1]

        return meta,meta_filter




def np_to_tensor(meta_filter,img_val):
    #filter img_val
    tile_index = meta_filter.index.tolist()
    img_val_filter=[img_val[i] for i in tile_index]
    #convert to tf.tensor
    img_val_filter_np=np.array(img_val_filter)
    img_tensor=torch.from_numpy(img_val_filter_np.transpose((0,3,1,2)))
    #print(img_tensor.shape)
    return img_tensor

'''
Calculate adjacent patches use X/Y coordinates
'''
from scipy.spatial import distance_matrix, minkowski_distance, distance
def calcADJ(coord, k=8, distanceType='euclidean', pruneTag='NA'):
    
    spatialMatrix=coord#.cpu().numpy()
    nodes=spatialMatrix.shape[0] 
    Adj=torch.zeros((nodes,nodes))
    for i in np.arange(spatialMatrix.shape[0]): 
        tmp=spatialMatrix[i,:].reshape(1,-1) 
        distMat = distance.cdist(tmp,spatialMatrix, distanceType) #eucilidean distance, manhatten may also be ok
        if k == 0:
            k = spatialMatrix.shape[0]-1
        res = distMat.argsort()[:k+1] 
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
        for j in np.arange(1,k+1):
            # No prune
            if pruneTag == 'NA':
                Adj[i][res[0][j]]=1.0
            elif pruneTag == 'STD':
                if distMat[0,res[0][j]]<=boundary:
                    Adj[i][res[0][j]]=1.0
            # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
            elif pruneTag == 'Grid':
                if distMat[0,res[0][j]]<=2**(1/2):
                    Adj[i][res[0][j]]=1.0
    return Adj
