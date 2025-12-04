from utils import get_config, get_images
from benchmark import xfeat_bm, utils_bm, roma_bm    
from pathlib import Path
from benchmark.clustering_2 import AGLP_clustering_2, proj_hdbscan_2, dissim_hdbscan_2
import numpy as np


if __name__ == '__main__':

    #### Get Config
    cfg = get_config()

    #### Extract Similarities
    ds = cfg['Dataset']
    treasure_name = cfg['Treasure']
    OUTDIR = cfg['Outdir']

    #### Init paths or vars
    # OUTDIR = f"SIMILARITIES/{treasure_name}/{ds}"
    FNAME = f"{OUTDIR}/matches.npy"
    DIST = f"{OUTDIR}/distances.npy"
    CMAP_PRED = f"{OUTDIR}/cmap_pred.txt"
    DIE_STUDIE = f"{OUTDIR}/die_studie.txt"


    #### Clustering
    sim = np.load(FNAME)
    dist = np.load(DIST)
    partition = []
    if cfg['Clustering'] == 'AGLP':
        partition = AGLP_clustering_2(sim, dist)
    # elif cfg['Clustering'] == 'HDBSCAN-Proj':
    #     partition = proj_hdbscan_2(sim, dist)
    # elif cfg['Clustering'] == 'HDBSCAN-Dissim':
    #     partition = dissim_hdbscan_2(dist)
    else:
        raise ValueError('Wrong Clustering selected. Must be in : AGLP ')
        # raise ValueError('Wrong Clustering selected. Must be in : AGLP | HDBSCAN-Dissim | HDBSCAN-Proj ')

    print("[RUN CLUSTERING] Final partition:")
    print(f"partition : {partition}")
    print(partition, partition.dtype, partition)
    np.savetxt(DIE_STUDIE, partition, fmt="%i") 
    print(f"[RUN CLUSTERING] Clustering results saved to {DIE_STUDIE}")
    np.savetxt(CMAP_PRED, partition, fmt="%i") 
    print(f"[RUN CLUSTERING] Clustering results saved to {CMAP_PRED}")


