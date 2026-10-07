from utils import get_config, get_images
from benchmark import xfeat_bm, utils_bm, roma_bm    
from pathlib import Path
# from benchmark.clustering_2 import AGLP_clustering_2, ALGP_clustering_BM, ALGP_clustering_percentile_BM, ConnectedComponents_BM, proj_hdbscan_BM, dissim_hdbscan_BM#, proj_hdbscan_2, dissim_hdbscan_2
from benchmark.clustering_cornet import ALGP_clustering_BM, ConnectedComponents_BM, proj_hdbscan_BM, dissim_hdbscan_BM
import numpy as np


if __name__ == '__main__':

      _dump = False
      _function = "[cornet_bm] [run_clustering.py] "

      #### Get Config
      cfg = get_config()

      #### Extract Similarities
      ds = cfg['Dataset'] # not used here, but keeping for consistency
      treasure_name = cfg['Treasure'] # not used here, but keeping for consistency
      INDIR = cfg['Indir']
      OUTDIR = cfg['Outdir']
      program = cfg['Clustering']


      # print(f"[RUN CLUSTERING] Program: {program}")

      
      print(f"{_function} Start... Program: {program} ...")
      # # TOTO PAT : ajouer un filtre sur cfg['Clustering']

      # print(f"[RUN CLUSTERING] INDIR: {INDIR}")
      # print(f"[RUN CLUSTERING] OUTDIR: {OUTDIR}")

      #### Init paths or vars
      SIM_MATRIX = str(Path(INDIR, "matches.npy"))
      print(f"{_function} SIM_MATRIX: {SIM_MATRIX}")
      
      DIST_MATRIX = str(Path(INDIR, "distances.npy"))
      print(f"{_function} DIST_MATRIX: {DIST_MATRIX}")

      CMAP_PRED = str(Path(OUTDIR, "cmap_pred.txt"))
      print(f"{_function} CMAP_PRED: {CMAP_PRED}")

      DIE_STUDIE = str(Path(OUTDIR, "die_studie.txt"))
      print(f"{_function} DIE_STUDIE: {DIE_STUDIE}")

      

      #### Clustering
      sim = np.load(SIM_MATRIX)
      dmat = np.load(DIST_MATRIX)

      if _dump:
            print(f"{_function} sim  = {sim}")
            print(f"{_function} dist = {dmat}")

      partition = []
      #     if cfg['Clustering'] == 'AGLP':
      #         print('[RUN CLUSTERING] AGLP clustering ...')
      #         partition = AGLP_clustering_2(sim, dist)
      #         print('[RUN CLUSTERING] AGLP clustering ... Done.')
      #     elif cfg['Clustering'] == 'HDBSCAN-Proj':
      #          partition = proj_hdbscan_BM(sim,dist)
      #     elif cfg['Clustering'] == 'HDBSCAN-Dissim':
      #          partition = dissim_hdbscan_BM(dist)
      #     elif cfg['Clustering'] == 'AGLP BM':
      #          partition = ALGP_clustering_BM(sim, dist)
      #     elif cfg['Clustering'] == 'AGLP Percentile BM':
      #          partition = ALGP_clustering_percentile_BM(sim, dist)
      #     elif cfg['Clustering'] == 'Connected Components':
      #          partition = ConnectedComponents_BM(sim, dist)
      #     # elif cfg['Clustering'] == 'HDBSCAN-Proj':
      #     #     partition = proj_hdbscan_2(sim, dist)
      #     # elif cfg['Clustering'] == 'HDBSCAN-Dissim':
      #     #     partition = dissim_hdbscan_2(dist)

      if program =='AGLP': 
            partition= ALGP_clustering_BM(sim, dmat) ### CORNET
      elif program =='ConnectedComponents':
            partition = ConnectedComponents_BM(sim, dmat) ### CORNET
      elif program == 'HDBSCAN-Proj':
            partition = proj_hdbscan_BM(sim, dmat) ### CORNET
      elif program =='HDBSCAN-Dissim':
            partition = dissim_hdbscan_BM(dmat) ### CORNET

      else:
            raise ValueError(f'Wrong Clustering selected. Must be in : AGLP | ConnectedComponents | HDBSCAN-Proj | HDBSCAN-Dissim. Got: {program}')

      if _dump:
            print(f"{_function} Final partition: {partition}")

      # create path if not exists
      Path(OUTDIR).mkdir(parents=True, exist_ok=True)

      # save "die_studie.txt"
      np.savetxt(DIE_STUDIE, partition, fmt="%i") 

      if _dump:
            print(f"{_function} Clustering results saved to {DIE_STUDIE}")

      # save "cmap_pred.txt"
      np.savetxt(CMAP_PRED, partition, fmt="%i") 
      
      if _dump:
            print(f"{_function} Clustering results saved to {CMAP_PRED}")

      # print partition details
      if _dump:
            print(f"{_function} Partition details: {partition}")
            print(f"{_function} partition.dtype: {partition.dtype}")


      print(f"{_function} END... Program: {program} ")
