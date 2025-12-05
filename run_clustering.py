from utils import get_config, get_images
from benchmark import xfeat_bm, utils_bm, roma_bm    
from pathlib import Path
from benchmark.clustering_2 import AGLP_clustering_2, AGLP_clustering_NICO, proj_hdbscan_2, dissim_hdbscan_2
import numpy as np


if __name__ == '__main__':

    #### Get Config
    cfg = get_config()

    #### Extract Similarities
    ds = cfg['Dataset'] # not used here, but keeping for consistency
    treasure_name = cfg['Treasure'] # not used here, but keeping for consistency
    INDIR = cfg['Indir']
    OUTDIR = cfg['Outdir']

    #### Init paths or vars
    SIM_MATRIX = f"{INDIR}/matches.npy"
    DIST_MATRIX = f"{INDIR}/distances.npy"
    CMAP_PRED = f"{OUTDIR}/cmap_pred.txt"
    DIE_STUDIE = f"{OUTDIR}/die_studie.txt"

    #### Clustering
    sim = np.load(SIM_MATRIX)
    dist = np.load(DIST_MATRIX)
    # print("sim  =", sim ) 
    # print("dist =", dist )
    
    partition = []
    if cfg['Clustering'] == 'AGLP':
        print('[RUN CLUSTERING] AGLP clustering ...')
        # partition = AGLP_clustering_2(sim, dist)
        partition = AGLP_clustering_NICO(sim, dist)
        print('[RUN CLUSTERING] AGLP clustering ... Done.')
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
