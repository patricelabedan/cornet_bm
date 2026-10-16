import numpy as np
import os
import torch
from skimage import io
from pathlib import Path
from romatch.models.model_zoo import roma_outdoor


def _poi_couple_file_path(dir_poi, file_i, file_j):
    name_i = os.path.basename(file_i)
    name_j = os.path.basename(file_j)
    return str(Path(dir_poi, f"poi_couples____{name_i}____{name_j}____.npy"))


def _legacy_poi_couple_file_path(dir_poi, file_i, file_j):
    name_i = os.path.basename(file_i)
    name_j = os.path.basename(file_j)
    return str(Path(dir_poi, f"poi_couples____{name_i}____{name_j}.npy"))


def save_poi_couples(files, threshold=0.9, out_dir='similarities/poi_couples'):
    """
    Pour chaque couple d'images, extrait les points d'intérêt RoMa et les sauvegarde dans un fichier .npy.
    Format: poi_couples____{name_i}____{name_j}____.npy (extensions included).
    Image basenames must be unique within the dataset; hashes are not used.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roma_model = roma_outdoor(device=device)

    # Pré-chargement des images
    IMAGES = [io.imread(im) for im in files]

    for i, im1 in enumerate(IMAGES):
        for j, im2 in enumerate(IMAGES):
            # RoMa retourne (matches, certainty)
            matches, certainty = roma_model.match(im1, im2, device=device)
            # On ne garde que les matches dont la certitude dépasse le seuil
            mask = (certainty > threshold)
            points1 = matches[0][mask]
            points2 = matches[1][mask]
            matches_list = [points1, points2]
            fname = _poi_couple_file_path(out_dir, files[i], files[j])
            np.save(fname, matches_list)
			



# def build_similarity_matrix_from_poi_couples(files, 
#                                              out_dir='similarities/poi_couples', 
#                                              fname='similarities/matches.npy'):
def build_similarity_matrix_from_poi_couples(files, 
                                             filtering, 
                                             dir_poi, 
                                             fname):
    """
    Construit la matrice de similarités à partir des fichiers poi_couples____...____.npy (RoMa).
    Chaque valeur est le nombre de points d'intérêt matchés pour chaque paire d'images.
    """
    _dump = False
    _function = "_build_similarity_matrix_from_poi_couples_"

    if _dump:
        print(f"[{_function}] START...")
        print(f"[{_function}] dir_poi = {dir_poi}")
        print(f"[{_function}] fname = {fname}")
        print(f"[{_function}] filtering = {filtering}")
        print(f"[{_function}] number of files = {len(files)}")
        # print(f"[{_function}] files = {files}")


    similarities = np.zeros((len(files), len(files)))
    for i, file_i in enumerate(files):
        for j, file_j in enumerate(files):
            poi_file = _poi_couple_file_path(dir_poi, file_i, file_j)
            if not os.path.exists(poi_file):
                poi_file = _legacy_poi_couple_file_path(dir_poi, file_i, file_j)
            if not os.path.exists(poi_file):
                print(f"[{_function}] POI file does not exist: {poi_file}")
                similarities[i][j] = 0
                continue
            matches_list = np.load(poi_file, allow_pickle=True)
            # matches_list[0] et matches_list[1] sont les points matchés
            similarities[i][j] = len(matches_list[0])

    if _dump:
        print(f"[{_function}] Similarity matrix saved in fname = {fname}...")

    np.save(fname, similarities)

    if _dump:
        print(f"[{_function}] Similarity matrix saved in fname = {fname}... OK")

