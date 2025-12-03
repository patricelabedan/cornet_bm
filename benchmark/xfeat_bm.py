import numpy as np
from skimage import io
import torch
from extract_features.xfeat_cache import XFeat
import tqdm
import pandas as pd
import time
import cv2
from pathlib import Path
import os
import glob
from benchmark.utils_bm import save_distance_matrix_to_csv, sort_csv_by_distance



def clean_poi_couples_directory(out_dir='similarities/poi_couples'):
    """
    Supprime tous les fichiers dans le répertoire des poi_couples.
    Utile pour libérer de l'espace disque après avoir construit la matrice de similarités.
    """
    files = glob.glob(f"{out_dir}/poi_couples____*.npy")
    for f in files:
        os.remove(f)
    print(f"Tous les fichiers dans {out_dir} ont été supprimés.")




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




def create_distance_matrix(path_matrix_sim, 
                           path_matrix_dist,
                           final_dest):
    """
    Creates a distance matrix from a similarity matrix and saves it to a .npy file.
    """
    if not os.path.exists(path_matrix_sim):
        raise FileNotFoundError(f"Similarity matrix does not exist : '{path_matrix_sim}'")
    sim = np.load(path_matrix_sim)
    dmat = sim.max() - sim
    np.fill_diagonal(dmat, 0)
    np.save(path_matrix_dist, dmat)

    parent_folder = Path(path_matrix_dist).parent

    path_result_distances_not_sorted = str(Path(parent_folder, "results_distance_not_sorted.csv"))
    path_result_distances = str(Path(parent_folder, "results_distance.csv"))

    path_coin_list_txt = str(Path(final_dest, 'files_list.txt'))

    save_distance_matrix_to_csv(path_matrix_dist, 
                                path_coin_list_txt, 
                                path_result_distances_not_sorted)

    sort_csv_by_distance(path_result_distances_not_sorted, 
                         path_result_distances)






def build_similarity_matrix_from_poi_couples(files, 
                                             filtering, 
                                             dir_poi, 
                                             fname):
    """
    Construit la matrice de similarités à partir des fichiers poi_couples____...____.npy
    """
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
