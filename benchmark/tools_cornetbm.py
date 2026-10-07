from pathlib import Path
import numpy as np
import sys
import os
import shutil
import json
import datetime
import time
import pandas as pd

# from benchmark.metrics import computing_metrics

# sys.path.append(os.path.join(os.path.dirname(__file__), "benchmark"))

# from Src2.benchmark import get_config_clustering
# from Src2.clustering import AGGLOMERATIVE_Clustering 
# from Src2.clustering_ADLC import ADLC_Juillac
# from Src2.clustering_sklearn import OPTICS_BM, HDBSCAN_BM, AgglomerativeClustering_BM, AgglomerativeClustering_sil_BM, AgglomerativeClustering_sil_single_BM, AgglomerativeCLustering_umap_BM, AffinityPropagation_BM, AffinityPropagation_proj_BM, Birch_BM, AgglomerativeCLustering_MDS_BM, AgglomerativeClustering_simul_BM
# from Src2.partition_arrangement import SklearnNoiseArrangement, NamedPartition




def loadGroundTruthDS(path_treasure, ds):
    '''
    Load GroundTruth.csv for a given dataset and treasure, and return the list of links (pairs of coins that are linked).
    '''

    _function = "_loadGroundTruthDS_"
    _dump = False

    if _dump:
        print(f"{_function} Loading GroundTruth.csv in ### {path_treasure} ###...")
        print(f"{_function} path_treasure : {path_treasure}")
        print(f"{_function} ds : {ds}")

    path_ground_truth = str(Path(path_treasure, 'GroundTruth.csv'))
    if _dump:
        print(f"{_function} path_ground_truth : {path_ground_truth}")

    ground_truths = pd.read_csv(str(path_ground_truth), sep=";", header=None)
    if _dump:
        print(f"{_function} ground_truths : {ground_truths}")

    ground_truth = ground_truths.loc[ground_truths[0] == ds]
    if _dump:
        print(f"{_function} ground_truth : {ground_truth}")

    links = ground_truth[[1,2]].values.tolist()
    if _dump:
        print(f"{_function} links : {links}")        

    return links




def defineGroundTruthMatrix(ref_pic_list, links):
    '''
    Define the GroundTruth matrix for a given list of reference pictures and links.
    '''

    _function = "_defineGroundTruthMatrix_"
    _dump = False

    nb = len(ref_pic_list)
    if _dump:
        print(f"{_function} nb coins : {nb}")

    ref_pic_stems = [Path(p).stem for p in ref_pic_list]

    G = np.full((nb,nb), False)
    for l in links:
        c1,c2 = ref_pic_stems.index(Path(l[0]).stem),ref_pic_stems.index(Path(l[1]).stem)
        G[min(c1,c2),max(c1,c2)] = True
    
    if _dump:
        print(f"{_function} G : {G}")


    return G





# def removeFilesInDir(path_dir, files_to_remove):
#     print(f"Removing specified files in directory: {path_dir}...")
#     for file in files_to_remove:
#         file_path = os.path.join(path_dir, file)
#         if os.path.exists(file_path):
#             print(f"Removing file: {file_path}")
#             os.remove(file_path)
#     print(f"Removing specified files in directory: {path_dir}... OK")




def copier_fichiers(source_dir, dest_dir, liste_fichiers):
    """
    Copie les fichiers spécifiés de source_dir vers dest_dir.

    Args:
        source_dir (str): Chemin du dossier source.
        dest_dir (str): Chemin du dossier destination.
        liste_fichiers (list): Liste des noms de fichiers à copier.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for nom_fichier in liste_fichiers:
        chemin_source = os.path.join(source_dir, nom_fichier)
        chemin_dest = os.path.join(dest_dir, nom_fichier)
        if os.path.isfile(chemin_source):
            shutil.copy2(chemin_source, chemin_dest)
        else:
            print(f"Fichier non trouvé : {chemin_source}")






def computeG(path_treasure, ds, path_ref_pic_list):

    _function = "_computeG_"
    _dump = False

    ref_pic_list = np.loadtxt(path_ref_pic_list, dtype=str)
    if _dump:
        print(f"[{_function}] Partition loaded from ref_pic_list : {ref_pic_list}")

    links = loadGroundTruthDS(path_treasure, ds)
    nbCoins = len(ref_pic_list)

    if _dump:
        print(f"[{_function}] ***** Display vars:")
        print(f"[{_function}] nbCoins : {nbCoins}")
        print(f"[{_function}] ref_pic_list : {ref_pic_list}")

    # Convert Path to str
    ref_pic_list = [str(p) for p in ref_pic_list]

    g = defineGroundTruthMatrix(ref_pic_list, links)

    return g











def save_metrics(path_cluster_ds, 
                 path_project, 
                 metrics,
                 treasure,
                 family_cluster,
                 algo_cluster):

    _dump = False
    _function = "_save_metrics_"

    b_ResultsFolder = False

    # Sauvegarder dans le dossier similarities du dataset
    metrics_path = Path(path_cluster_ds, family_cluster, algo_cluster, 'metrics.json')
    # print(f"Saving metrics to: {metrics_path}")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    if _dump:
        print(f"[{_function}] Metrics saved to: {metrics_path}")


    if b_ResultsFolder:

        # Copie avec timestamp dans le dossier RESULTS
        timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")        
        results_dir = Path(path_project, 'Datasets', 'RESULTS')
        results_dir.mkdir(exist_ok=True)

        ds = str(Path(path_cluster_ds).name)

        file_name_metrics = f"metrics_{treasure}_{ds}_{family_cluster}_{algo_cluster}_{timestamp}.json"
        metrics_timestamped_path = str(Path(results_dir, file_name_metrics))
        with open(metrics_timestamped_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        if _dump:
            print(f"[{_function}] Timestamped metrics saved to: {metrics_timestamped_path}")


