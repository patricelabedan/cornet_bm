import numpy as np
from skimage import io
import torch
from extract_features.xfeat_cache import XFeat
import tqdm
import cv2
import os
# from utils_bm import saveCoinList
from pathlib import Path
from benchmark import utils_bm


def _poi_couple_file_path(dir_poi, file_i, file_j):
    name_i = os.path.basename(file_i)
    name_j = os.path.basename(file_j)
    return str(Path(dir_poi, f"poi_couples____{name_i}____{name_j}____.npy"))


def save_poi_couples(files, top_k=5000, out_dir='similarities/poi_couples'):
    """
    Pour chaque couple d'images, extrait les points d'intérêt XFeat et les sauvegarde dans un fichier .npy.
    Format: poi_couples____{name_i}____{name_j}____.npy (extensions included).
    Image basenames must be unique within the dataset; hashes are not used.
    """
    os.makedirs(out_dir, exist_ok=True)
    IMAGES = [io.imread(im) for im in files]
    im_list = []
    for im in IMAGES:
        if len(im.shape) == 3:
            im_list.append(torch.tensor(im.transpose(2,0,1)))
        else:
            im_list.append(torch.tensor(im[None,:,:]))
    xfeat = XFeat(top_k=top_k)
    xfeat.cache_feats(im_list)

    for i in tqdm.tqdm(range(len(files)), desc='POI Couples'):
        for j in range(len(files)):
            matches_list = xfeat.match_xfeat_star_from_cache(i, j)
            # matches_list est typiquement une liste de deux arrays (points d'intérêt dans chaque image)
            fname = _poi_couple_file_path(out_dir, files[i], files[j])
            np.save(fname, matches_list)





def build_similarity_matrix_from_poi_couples(files, 
                                             filtering, 
                                             dir_poi, 
                                             fname):
    """
    Construit la matrice de similarités à partir des fichiers poi_couples____...____.npy
    """
    _dump = True
    _function = "_build_similarity_matrix_from_poi_couples_"

    if _dump:
        print(f"[{_function}] START...")
        print(f"[{_function}] dir_poi = {dir_poi}")
        print(f"[{_function}] fname = {fname}")
        print(f"[{_function}] filtering = {filtering}")
        print(f"[{_function}] number of files = {len(files)}")
        # print(f"[{_function}] files = {files}")

    # print(files)
    similarities = np.zeros((len(files), len(files)))

    for i in tqdm.tqdm(range(len(files)), desc='Build Similarity Matrix'):
        for j in range(len(files)):
            # if _dump:
            #     print(f"[{_function}] Processing pair: {files[i]}, {files[j]}")
            poi_file = _poi_couple_file_path(dir_poi, files[i], files[j])
            if not os.path.exists(poi_file):
                print(f"[{_function}] POI file does not exist: {poi_file}")
                similarities[i][j] = 0
                continue
            matches_list = np.load(poi_file, allow_pickle=True)
            if not filtering:
                similarities[i][j] = len(matches_list[0])
            else:
                if len(matches_list[0]) < 4 or len(matches_list[1]) < 4:
                    similarities[i][j] = 0
                    continue
                _, mask = cv2.findHomography(
                    matches_list[0],
                    matches_list[1],
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=8,
                )
                if mask is not None:
                    similarities[i][j] = mask.sum()
                else:
                    similarities[i][j] = 0
    if _dump:
        print(f"[{_function}] Similarity matrix saved in fname = {fname}...")
    
    np.save(fname, similarities)

    if _dump:
        print(f"[{_function}] Similarity matrix saved in fname = {fname}... OK")

    # Also save the list of files used to build the matrix
    with open(fname.replace('.npy', '_files.txt'), 'w', encoding='utf-8') as f:
        for file in files:
            f.write(f"{file}\n")
        if _dump:
            print(f"[{_function}] Files list saved in {fname.replace('.npy', '_files.txt')}")

    filename = str(Path(str(Path(dir_poi).parent), 'files_list.txt'))

    if _dump:
        print(f"[{_function}] Saving coin list in {filename} ...")

    utils_bm.saveCoinList(filename, files)

    if _dump:
        print(f"[{_function}] Saving coin list in {filename} ... OK")







def build_similarity_matrix_from_poi_couples_accadil(files, 
                                             filtering, 
                                             dir_poi, 
                                             fname):
    """
    Construit la matrice de similarités à partir des fichiers poi_couples____...____.npy 
    générés par le matching accadil (cad que dans un sens A-B).
    Donc il n'y a pas tous les fichiers poi_couples possibles (pas les diagonales ni les couples inverses).
    (par ex : pour 4 monnaies, il n'y aura pas 16 fichiers poi_couples, mais seulement 6 fichiers) (=n*(n-1)/2)
    """
    _dump = True
    _function = "_build_similarity_matrix_from_poi_couples_accadil_"

    if _dump:
        print(f"[{_function}] START...")
        print(f"[{_function}] dir_poi = {dir_poi}")
        print(f"[{_function}] fname = {fname}")
        print(f"[{_function}] filtering = {filtering}")
        print(f"[{_function}] number of files = {len(files)}")
        # print(f"[{_function}] files = {files}")

    # print(files)
    similarities = np.zeros((len(files), len(files)))

    for i in tqdm.tqdm(range(len(files)), desc='Build Similarity Matrix'):
        for j in range(len(files)):
            # if _dump:
            #     print(f"[{_function}] Processing pair: {files[i]}, {files[j]}")
            if files[i] == files[j]:
                similarities[i][j] = 0 #ou si i==j, pareil
                continue    

            poi_file_A = _poi_couple_file_path(dir_poi, files[i], files[j])
            poi_file_B = _poi_couple_file_path(dir_poi, files[j], files[i])

            if not os.path.exists(poi_file_A):
                if not os.path.exists(poi_file_B):
                    print(f"[{_function}] POI files do not exist: {poi_file_A}, {poi_file_B}... Exiting")
                    exit()
                else:
                    poi_file = poi_file_B
            else:
                if not os.path.exists(poi_file_B):
                    poi_file = poi_file_A
                else:
                    print(f"[{_function}] POI file supposed to not exist: {poi_file_B}... Exiting")
                    exit()

            matches_list = np.load(poi_file, allow_pickle=True)

            if not filtering:
                similarities[i][j] = len(matches_list[0])
                similarities[j][i] = len(matches_list[0]) # symétrique
            else:
                if len(matches_list[0]) < 4 or len(matches_list[1]) < 4:
                    similarities[i][j] = 0
                    similarities[j][i] = 0
                    continue
                _, mask = cv2.findHomography(
                    matches_list[0],
                    matches_list[1],
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=8,
                )
                if mask is not None:
                    similarities[i][j] = mask.sum()
                    similarities[j][i] = mask.sum() # symétrique
                else:
                    similarities[i][j] = 0
                    similarities[j][i] = 0 # symétrique

    if _dump:
        print(f"[{_function}] Similarity matrix saved in fname = {fname}...")
    
    np.save(fname, similarities)

    if _dump:
        print(f"[{_function}] Similarity matrix saved in fname = {fname}... OK")

    # Also save the list of files used to build the matrix
    with open(fname.replace('.npy', '_files.txt'), 'w', encoding='utf-8') as f:
        for file in files:
            f.write(f"{file}\n")
        if _dump:
            print(f"[{_function}] Files list saved in {fname.replace('.npy', '_files.txt')}")

    filename = str(Path(str(Path(dir_poi).parent), 'files_list.txt'))

    if _dump:
        print(f"[{_function}] Saving coin list in {filename} ...")

    utils_bm.saveCoinList(filename, files)

    if _dump:
        print(f"[{_function}] Saving coin list in {filename} ... OK")
