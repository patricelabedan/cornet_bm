import numpy as np
import os
import torch
from pathlib import Path
from romatch.models.model_zoo import roma_outdoor
from benchmark import utils_bm


def _poi_couple_file_path(dir_poi, file_i, file_j):
    name_i = os.path.basename(file_i)
    name_j = os.path.basename(file_j)
    return str(Path(dir_poi, f"poi_couples____{name_i}____{name_j}____.npy"))


def save_poi_couples(files, threshold=0.9, out_dir='similarities/poi_couples'):
    """
    Pour chaque couple d'images, extrait les points d'intérêt RoMa et les sauvegarde dans un fichier .npy.
    Le paramètre threshold est appliqué ici sur la certitude RoMa pour conserver uniquement
    les correspondances suffisamment fiables.
    Format: poi_couples____{name_i}____{name_j}____.npy (extensions included).
    Image basenames must be unique within the dataset; hashes are not used.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roma_model = roma_outdoor(device=device)
    for i, file_i in enumerate(files):
        for j, file_j in enumerate(files):
            # RoMa retourne un champ dense de correspondances (xA,yA,xB,yB) et une certitude
            matches, certainty = roma_model.match(file_i, file_j, device=device)
            # On aplatit pour aligner correctement certitudes et correspondances
            matches_flat = matches.reshape(-1, 4)
            certainty_flat = certainty.reshape(-1)
            mask = certainty_flat > threshold
            good_matches = matches_flat[mask]
            points1 = good_matches[:, :2].detach().cpu().numpy()
            points2 = good_matches[:, 2:].detach().cpu().numpy()
            matches_list = [points1, points2]
            fname = _poi_couple_file_path(out_dir, files[i], files[j])
            np.save(fname, matches_list)


def build_similarity_matrix_from_poi_couples(files, 
                                             dir_poi, 
                                             fname):
    """
    Construit la matrice de similarités à partir des fichiers poi_couples____...____.npy (RoMa).
    Chaque valeur est le nombre de points d'intérêt matchés pour chaque paire d'images.
    Les correspondances ont déjà été filtrées en amont dans save_poi_couples via threshold.
    """
    _dump = False
    _function = "_build_similarity_matrix_from_poi_couples_"

    if _dump:
        print(f"[{_function}] START...")
        print(f"[{_function}] dir_poi = {dir_poi}")
        print(f"[{_function}] fname = {fname}")
        print(f"[{_function}] number of files = {len(files)}")
        # print(f"[{_function}] files = {files}")


    similarities = np.zeros((len(files), len(files)))
    for i, file_i in enumerate(files):
        for j, file_j in enumerate(files):
            poi_file = _poi_couple_file_path(dir_poi, file_i, file_j)
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
                                                     dir_poi,
                                                     fname):
    """
    Construit la matrice de similarités à partir des fichiers poi_couples____...____.npy
    générés par le matching accadil (cad que dans un sens A-B).
    Donc il n'y a pas tous les fichiers poi_couples possibles (pas les diagonales ni les couples inverses).
    (par ex : pour 4 monnaies, il n'y aura pas 16 fichiers poi_couples, mais seulement 6 fichiers) (=n*(n-1)/2)
    """
    _dump = False
    _function = "_build_similarity_matrix_from_poi_couples_accadil_"

    if _dump:
        print(f"[{_function}] START...")
        print(f"[{_function}] dir_poi = {dir_poi}")
        print(f"[{_function}] fname = {fname}")
        print(f"[{_function}] number of files = {len(files)}")

    similarities = np.zeros((len(files), len(files)))

    for i, file_i in enumerate(files):
        for j, file_j in enumerate(files):
            if file_i == file_j:
                similarities[i][j] = 0
                continue

            poi_file_a = _poi_couple_file_path(dir_poi, file_i, file_j)
            poi_file_b = _poi_couple_file_path(dir_poi, file_j, file_i)

            if not os.path.exists(poi_file_a):
                if not os.path.exists(poi_file_b):
                    raise FileNotFoundError(
                        f"[{_function}] POI files do not exist: {poi_file_a}, {poi_file_b}"
                    )
                poi_file = poi_file_b
            else:
                if not os.path.exists(poi_file_b):
                    poi_file = poi_file_a
                else:
                    raise ValueError(
                        f"[{_function}] Both POI files exist but only one is expected: {poi_file_a}, {poi_file_b}"
                    )

            matches_list = np.load(poi_file, allow_pickle=True)
            similarities[i][j] = len(matches_list[0])
            similarities[j][i] = len(matches_list[0])

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
