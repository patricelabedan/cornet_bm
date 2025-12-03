import numpy as np
from skimage import io
import torch
from extract_features.xfeat_cache import XFeat
import tqdm
import pandas as pd
import time
import cv2
from pathlib import Path


def save_matches(files, top_k=5000, filtering=True, fname='similarities/matches.npy'):
    """
    Extract pairwise XFeat matches from a collection.

    Parameters:
        - files (list(str)) : List of image paths
        - top_k (int) : Maximum number of regions to match through XFeat
        - fname (str) : Path to save the results, as a `.npy` file
    """

    start_time = time.time()

    IMAGES = [io.imread(im) for im in files]
    im_list = []
    # if len(IMAGES[0].shape) == 3:
    #     im_list = [torch.tensor(im.transpose(2,0,1)) for im in IMAGES]
    # else:
    #     im_list = [torch.tensor(im[None,:,:]) for im in IMAGES]
    # previous code is fine if __all__ images have the same size as the first one
    for im in IMAGES:
        if len(im.shape) == 3:
            im_list.append(torch.tensor(im.transpose(2,0,1)))
        else:
            im_list.append(torch.tensor(im[None,:,:]))

    xfeat = XFeat(top_k=top_k)
    xfeat.cache_feats(im_list)
    
    def matches_two_files(i1, i2):
        #x1 = torch.tensor(IMAGES[i1].transpose(2,0,1))
        #x2 = torch.tensor(IMAGES[i2].transpose(2,0,1))
        matches_list = xfeat.match_xfeat_star_from_cache(i1, i2)
        # if not filtering
        matches_list_np = np.array([matches_list[0],matches_list[1]])
        # name1 = files[i1].split('/')[-1].replace('.jpg','')
        # name2 = files[i2].split('/')[-1].replace('.jpg','')
        name1 = Path(files[i1]).name # .split('/')[-1].replace('.jpg','')
        name2 = Path(files[i2]).name        
        print("name1 = ", name1)
        print("name2 = ", name2)
        
        np.save('similarities/poi_couples____' + name1 + '____' + name2 + '____.npy',matches_list_np)
        # poi = np.load('similarities/poi_couples.npy')
        if not filtering: return len(matches_list[0])
        
        res, mask = cv2.findHomography(
            matches_list[0],
            matches_list[1],
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=8,
        )
        mask_bool = (mask>0)[:,0]
        # if filtering
        matches_list_np = np.zeros([2,mask.sum(),2])
        # IMG1
        matches_list_np[0,:,0] = matches_list[0][mask_bool,0]
        matches_list_np[0,:,1] = matches_list[0][mask_bool,1]
        # IMG2
        matches_list_np[1,:,0] = matches_list[1][mask_bool,0]
        matches_list_np[1,:,1] = matches_list[1][mask_bool,1]

        np.save('similarities/poi_couples____' + name1 + '____' + name2 + '____.npy',matches_list_np)       
        #poi = np.load('similarities/poi_couples.npy')
        return mask.sum()

    similarities = np.zeros((len(files), len(files)))

    for i in tqdm.tqdm(range(len(files)), desc='Matching'):
        for j in range(len(files)):
            similarities[i][j] = matches_two_files(i, j)
    np.save(fname, similarities)

    print(f'Elapsed : {(time.time() - start_time):4f}s')





