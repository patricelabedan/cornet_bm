
import glob
import numpy as np
import csv
# import shutil
import os
from pathlib import Path


def get_images_in_treasure_dataset(treasure, dataset):

    files = glob.glob(f"datasets/{treasure}/{dataset}/*.*")
    files = list(
        filter(
            lambda f: f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.gif') or f.endswith('.tiff'), 
            files
            )
        )
    return sorted(files)



def save_distance_matrix_to_csv(distance_matrix_path, 
                                ref_pic_list_path, 
                                output_csv_path):
    
    distances = np.load(distance_matrix_path)
    with open(ref_pic_list_path, mode='r', encoding='utf-8') as f:
        ref_pic_list = [line.strip() for line in f if line.strip()]
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['name1', 'name2', 'Distance'])
        n = len(ref_pic_list)
        for i in range(n):
            for j in range(i+1, n):
                writer.writerow([ref_pic_list[i], ref_pic_list[j], distances[i, j]])


def sort_csv_by_distance(input_csv_path, 
                         output_csv_path):
    
    with open(input_csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        rows = list(reader)
        rows.sort(key=lambda x: float(x['Distance']))

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(rows)





def getListImagesInFolder(path_folder):
    """
    Recherche récursive de tous les fichiers images dans path_folder et ses sous-dossiers.
    Retourne une liste de chemins relatifs (par rapport à path_folder).
    """
    _dump = False
    _function = "_getListImagesInFolder_"

    path_Folder = Path(path_folder)
    list_Images = []

    # Recherche récursive avec rglob
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff'):
        list_Images.extend(path_Folder.rglob(ext))

    # Conversion en chemins relatifs
    list_Images_rel = [img.relative_to(path_Folder) for img in list_Images]

    if _dump:
        print(f"[{_function}] list_Images_rel = {list_Images_rel}")

    return list_Images_rel




def saveCoinList(path_file, ref_pic_list):
    
    _function = "_saveCoinList_"
    _dump = False
    if _dump:
        print(f"[{_function}] START...")
        print(f"[{_function}] filename = {path_file}")
    with open(str(Path(path_file)), 'w', encoding='utf-8') as f:
        for file in ref_pic_list:
            f.write(f"{file}\n")
        print(f"Files list saved in {str(Path(path_file))}")
    if _dump:
        print(f"[{_function}] ... STOP")





def addTempFilesForGTComparison(path_matrix_dist, 
                                path_coin_list_txt):
    
    outdir = Path(path_matrix_dist).parent

    path_csv_not_sorted = str(Path(outdir, "results_distance_not_sorted.csv"))
    path_csv_sorted = str(Path(outdir, "results_distance.csv"))

    save_distance_matrix_to_csv(path_matrix_dist, 
                                path_coin_list_txt, 
                                path_csv_not_sorted)
    
    sort_csv_by_distance(path_csv_not_sorted, 
                         path_csv_sorted)
    




def create_distance_matrix(path_matrix_sim, 
                           path_matrix_dist,
                           path_coin_list_txt):
    """
    Creates a distance matrix from a similarity matrix and saves it to a .npy file.
    """
    if not os.path.exists(path_matrix_sim):
        raise FileNotFoundError(f"Similarity matrix does not exist : '{path_matrix_sim}'")
    sim = np.load(path_matrix_sim)
    dmat = sim.max() - sim
    np.fill_diagonal(dmat, 0)
    np.save(path_matrix_dist, dmat)

    ### Save distances to CSV for groundtruth comparison
    addTempFilesForGTComparison(path_matrix_dist, path_coin_list_txt)






def clean_poi_couples_directory(out_dir='similarities/poi_couples'):
    """
    Supprime tous les fichiers dans le répertoire des poi_couples.
    Utile pour libérer de l'espace disque après avoir construit la matrice de similarités.
    """
    files = glob.glob(f"{out_dir}/poi_couples____*.npy")
    for f in files:
        os.remove(f)
    print(f"Tous les fichiers dans {out_dir} ont été supprimés.")

