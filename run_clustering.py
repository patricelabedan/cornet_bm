from utils import get_config, get_images
from benchmark import xfeat_bm, utils_bm, roma_bm    
from pathlib import Path
from benchmark.clustering_2 import AGLP_clustering_2, ALGP_clustering_BM, ALGP_clustering_percentile_BM, ConnectedComponents_clustering_BM, proj_hdbscan_BM, dissim_hdbscan_BM#, proj_hdbscan_2, dissim_hdbscan_2
import numpy as np


if __name__ == '__main__':

    #### Get Config
    cfg = get_config()

    #### Extract Similarities
    ds = cfg['Dataset'] # not used here, but keeping for consistency
    treasure_name = cfg['Treasure'] # not used here, but keeping for consistency
    INDIR = cfg['Indir']
    OUTDIR = cfg['Outdir']
    print(f"[RUN CLUSTERING] INDIR: {INDIR}")
    print(f"[RUN CLUSTERING] OUTDIR: {OUTDIR}")

    #### Init paths or vars
    SIM_MATRIX = f"{INDIR}/matches.npy"
    print(f"[RUN CLUSTERING] SIM_MATRIX: {SIM_MATRIX}")
    
    DIST_MATRIX = f"{INDIR}/distances.npy"
    print(f"[RUN CLUSTERING] DIST_MATRIX: {DIST_MATRIX}")

    CMAP_PRED = f"{OUTDIR}/cmap_pred.txt"
    print(f"[RUN CLUSTERING] CMAP_PRED: {CMAP_PRED}")

    DIE_STUDIE = f"{OUTDIR}/die_studie.txt"
    print(f"[RUN CLUSTERING] DIE_STUDIE: {DIE_STUDIE}")

    

    #### Clustering
    sim = np.load(SIM_MATRIX)
    dist = np.load(DIST_MATRIX)
    print(f"[RUN CLUSTERING] sim  = {sim}")
    print(f"[RUN CLUSTERING] dist = {dist}")
    
    partition = []
    if cfg['Clustering'] == 'AGLP':
        print('[RUN CLUSTERING] AGLP clustering ...')
        partition = AGLP_clustering_2(sim, dist)
        print('[RUN CLUSTERING] AGLP clustering ... Done.')
    elif cfg['Clustering'] == 'HDBSCAN-Proj':
         partition = proj_hdbscan_BM(sim,dist)
    elif cfg['Clustering'] == 'HDBSCAN-Dissim':
         partition = dissim_hdbscan_BM(dist)
    elif cfg['Clustering'] == 'AGLP BM':
         partition = ALGP_clustering_BM(sim, dist)
    elif cfg['Clustering'] == 'AGLP Percentile BM':
         partition = ALGP_clustering_percentile_BM(sim, dist)
    elif cfg['Clustering'] == 'Connected Components':
         partition = ConnectedComponents_clustering_BM(sim, dist)
    # elif cfg['Clustering'] == 'HDBSCAN-Proj':
    #     partition = proj_hdbscan_2(sim, dist)
    # elif cfg['Clustering'] == 'HDBSCAN-Dissim':
    #     partition = dissim_hdbscan_2(dist)
    else:
        raise ValueError('Wrong Clustering selected. Must be in : AGLP ')
        # raise ValueError('Wrong Clustering selected. Must be in : AGLP | HDBSCAN-Dissim | HDBSCAN-Proj ')

    print(f"[RUN CLUSTERING] Final partition: {partition}")

    # create path if not exists
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    # save "die_studie.txt"
    np.savetxt(DIE_STUDIE, partition, fmt="%i") 
    print(f"[RUN CLUSTERING] Clustering results saved to {DIE_STUDIE}")

    # save "cmap_pred.txt"
    np.savetxt(CMAP_PRED, partition, fmt="%i") 
    print(f"[RUN CLUSTERING] Clustering results saved to {CMAP_PRED}")

    # print partition details
    print(partition, partition.dtype, partition)
    print(f"[RUN CLUSTERING] Done.")
