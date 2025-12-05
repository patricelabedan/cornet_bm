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



def save_poi_couples(files, top_k=5000, out_dir='similarities/poi_couples'):
    """
    Pour chaque couple d'images, extrait les points d'intérêt XFeat et les sauvegarde dans un fichier .npy.
    Le nom du fichier est : poi_couples____{nom_image_i}____{nom_image_j}____.npy
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
            name_i = os.path.basename(files[i])
            name_j = os.path.basename(files[j])
            fname = f"{out_dir}/poi_couples____{name_i}____{name_j}____.npy"
            np.save(fname, matches_list)





def build_similarity_matrix_from_poi_couples(files, 
                                             filtering, 
                                             dir_poi, 
                                             fname):
    """
    Construit la matrice de similarités à partir des fichiers poi_couples____...____.npy
    """
    print(files)
    import os
    similarities = np.zeros((len(files), len(files)))
    for i in tqdm.tqdm(range(len(files)), desc='Build Similarity Matrix'):
        for j in range(len(files)):
            name_i = os.path.basename(files[i])
            name_j = os.path.basename(files[j])
            poi_file = f"{dir_poi}/poi_couples____{name_i}____{name_j}____.npy"
            if not os.path.exists(poi_file):
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
    np.save(fname, similarities)
    print(f"Similarity matrix saved in {fname}")

    # Also save the list of files used to build the matrix
    with open(fname.replace('.npy', '_files.txt'), 'w', encoding='utf-8') as f:
        for file in files:
            f.write(f"{file}\n")
        print(f"Files list saved in {fname.replace('.npy', '_files.txt')}")

    filename = str(Path(str(Path(dir_poi).parent), 'files_list.txt'))
    print(f"Saving coin list in {filename}")
    utils_bm.saveCoinList(filename, files)
