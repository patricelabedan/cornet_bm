import numpy as np
import os
import torch
from skimage import io
from romatch.models.model_zoo import roma_outdoor


def save_poi_couples(files, threshold=0.9, out_dir='similarities/poi_couples'):
    """
    Pour chaque couple d'images, extrait les points d'intérêt RoMa et les sauvegarde dans un fichier .npy.
    Le nom du fichier est : poi_couples____{nom_image_i}____{nom_image_j}.npy
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
            name_i = os.path.basename(files[i])
            name_j = os.path.basename(files[j])
            fname = f"{out_dir}/poi_couples____{name_i}____{name_j}.npy"
            np.save(fname, matches_list)
			



def build_similarity_matrix_from_poi_couples(files, 
                                             out_dir='similarities/poi_couples', 
                                             fname='similarities/matches.npy'):
    """
    Construit la matrice de similarités à partir des fichiers poi_couples____...____.npy (RoMa).
    Chaque valeur est le nombre de points d'intérêt matchés pour chaque paire d'images.
    """
    similarities = np.zeros((len(files), len(files)))
    for i, file_i in enumerate(files):
        for j, file_j in enumerate(files):
            name_i = os.path.basename(file_i)
            name_j = os.path.basename(file_j)
            poi_file = f"{out_dir}/poi_couples____{name_i}____{name_j}.npy"
            if not os.path.exists(poi_file):
                similarities[i][j] = 0
                continue
            matches_list = np.load(poi_file, allow_pickle=True)
            # matches_list[0] et matches_list[1] sont les points matchés
            similarities[i][j] = len(matches_list[0])
    np.save(fname, similarities)
    print(f"Matrice de similarités sauvegardée dans {fname}")



