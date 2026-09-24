import os
from utils import get_config, get_images
from benchmark import xfeat_bm, utils_bm, roma_bm    
from pathlib import Path
# from benchmark.clustering_2 import AGLP_clustering_2, ALGP_clustering_BM, ALGP_clustering_percentile_BM, ConnectedComponents_clustering_BM, proj_hdbscan_BM, dissim_hdbscan_BM#, proj_hdbscan_2, dissim_hdbscan_2
from benchmark.clustering_cornet import ALGP_clustering_BM, ConnectedComponents_clustering_BM, proj_hdbscan_BM, dissim_hdbscan_BM
import numpy as np
import time
from benchmark.metrics import computing_metrics
from benchmark.tools_cornetbm import copier_fichiers, save_metrics, computeG



def removeFilesInDir(path_dir, files_to_remove):
    print(f"Removing specified files in directory: {path_dir}...")
    for file in files_to_remove:
        file_path = os.path.join(path_dir, file)
        if os.path.exists(file_path):
            print(f"Removing file: {file_path}")
            os.remove(file_path)
    print(f"Removing specified files in directory: {path_dir}... OK")




def metrics_clustering(nbCoins, 
                       G, 
                       cmap_pred_path, 
                       algo_cluster_cornet,
                       path_project,
                       outdir_algo,
                       path_global_results_ds,
                       cfg):

    ARI, NMI, precision, recall, f1score, accuracy = computing_metrics(nbCoins, G, cmap_pred_path)


    algo_clustering_name = 'cornet' + '_' + algo_cluster_cornet

    # Créer le dictionnaire des métriques
    metrics = {
        "dataset": ds,
        "treasure": treasure_name,
        "preprocessing": cfg['preprocessing'],
        "matching_algo": cfg['matching_algo'],
        "distance": cfg['distance'],
        "clustering": algo_clustering_name,
        "metrics": {
            "ARI": float(ARI),
            "NMI": float(NMI),
            "precision": float(precision),
            "recall": float(recall),
            "f1score": float(f1score),
            "accuracy": float(accuracy)
        }
    }

    save_metrics(outdir_algo, path_project, metrics, treasure_name, 'cornet', algo_cluster_cornet)

    
#     path_global_clustering_final = str(Path(path_global_results_ds, 'CLUSTERING', 'cornet', algo_cluster_cornet))
    
   
#     copier_fichiers(outdir_algo, 
#                     path_global_clustering_final, 
#                     ["metrics.json"])    






if __name__ == '__main__':

      #### Get Config
      cfg = get_config()

      #### Extract Similarities
      ds = cfg['Dataset'] # not used here, but keeping for consistency
      treasure_name = cfg['Treasure'] # not used here, but keeping for consistency
      INDIR = cfg['Indir']
      OUTDIR = cfg['Outdir']
      path_coin_list_txt = cfg['path_coin_list_txt']
      path_abs_treasure_base = cfg["path_abs_treasure"]
      path_project = cfg["path_project"]
      path_global_results_ds = cfg["path_global_results_ds"]

      # pour la fonction metrics_clustering
      #   + cfg['preprocessing'],
      #   + cfg['matching_algo'],
      #   + cfg['distance'],


      G = computeG(path_abs_treasure_base, ds, path_coin_list_txt)


      ref_pic_list = np.loadtxt(path_coin_list_txt, dtype=str)
      nbCoins = len(ref_pic_list)

      print(f"ref_pic_list: {ref_pic_list}")
      print(f"nbCoins: {nbCoins}")


      print(f"[RUN CLUSTERING] INDIR: {INDIR}")
      print(f"[RUN CLUSTERING] OUTDIR: {OUTDIR}")

      #### Init paths or vars
      SIM_MATRIX = str(Path(INDIR, "matches.npy"))
      print(f"[RUN CLUSTERING] SIM_MATRIX: {SIM_MATRIX}")

      DIST_MATRIX = str(Path(INDIR, "distances.npy"))
      print(f"[RUN CLUSTERING] DIST_MATRIX: {DIST_MATRIX}")


      sim = np.load(SIM_MATRIX)
      dmat = np.load(DIST_MATRIX)
      print(f"[RUN CLUSTERING] sim  = {sim}")
      print(f"[RUN CLUSTERING] dist = {dmat}")

      

      algo_list = [
            'AGLP', 
            'ConnectedComponents_Clustering', 
            'HDBSCAN-Proj', 
            'HDBSCAN-Dissim'
      ]


      for algo in algo_list:

            print(f"[RUN CLUSTERING] algo: {algo}")

            OUTDIR_ALGO = str(Path(OUTDIR, 'cornet', algo))
            os.makedirs(OUTDIR_ALGO, exist_ok=True)

            CMAP_PRED = str(Path(OUTDIR_ALGO, "cmap_pred.txt"))
            print(f"[RUN CLUSTERING] CMAP_PRED: {CMAP_PRED}")

            DIE_STUDIE = str(Path(OUTDIR_ALGO, "die_studie.txt"))
            print(f"[RUN CLUSTERING] DIE_STUDIE: {DIE_STUDIE}")
           
            files_to_remove = ["cmap_pred.txt", "die_studie.txt", "metrics.json"]
            removeFilesInDir(OUTDIR_ALGO, files_to_remove)
            
            partition = []
  
            if algo == 'AGLP': 
                  partition= ALGP_clustering_BM(sim, dmat) ### CORNET
            elif algo == 'ConnectedComponents_Clustering':
                  partition = ConnectedComponents_clustering_BM(sim, dmat) ### CORNET
            elif algo == 'HDBSCAN-Proj':
                  partition = proj_hdbscan_BM(sim, dmat) ### CORNET
            elif algo == 'HDBSCAN-Dissim':
                  partition = dissim_hdbscan_BM(dmat) ### CORNET

            else:
                  raise ValueError(f'Wrong Clustering selected. Must be in : {algo_list} ')

            print(f"[RUN CLUSTERING] Final partition: {partition}")

            # create path if not exists
            Path(OUTDIR).mkdir(parents=True, exist_ok=True)

            # save "die_studie.txt"
            np.savetxt(DIE_STUDIE, partition, fmt="%i") 
            print(f"[RUN CLUSTERING] Clustering results saved to {DIE_STUDIE}")

            # save "cmap_pred.txt"
            np.savetxt(CMAP_PRED, partition, fmt="%i") 
            print(f"[RUN CLUSTERING] Clustering results saved to {CMAP_PRED}")

            # Attendre que le fichier 'cmap_pred.txt' soit bien écrit (sans limite de temps)
            start_time = time.time()
            last_print = -1
            while not os.path.exists(CMAP_PRED):
                  elapsed = int(time.time() - start_time)
                  if elapsed != last_print:
                        print(f"[WAIT] Attente de création de cmap_pred.txt depuis {elapsed} seconde(s)...")
                        last_print = elapsed
                  time.sleep(0.1)        


            # print partition details
            # print(partition, partition.dtype, partition)
            print(f"[RUN CLUSTERING] Done.")

            metrics_clustering(nbCoins, 
                              G, 
                              CMAP_PRED, 
                              algo,
                              path_project,
                              OUTDIR,
                              path_global_results_ds, # inside : + 'CLUSTERING', 'cornet', algo
                              cfg)


            path_global_clustering_final = str(Path(path_global_results_ds, 'CLUSTERING', 'cornet', algo))

            print(f"[RUN CLUSTERING] Files copied to {path_global_clustering_final}...")
            copier_fichiers(OUTDIR_ALGO, 
                            path_global_clustering_final, 
                            ["cmap_pred.txt", "die_studie.txt", "metrics.json"])            
            print(f"[RUN CLUSTERING] Files copied to {path_global_clustering_final}... OK")
            