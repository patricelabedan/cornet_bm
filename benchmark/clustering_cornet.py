from turtle import distance
from scipy import sparse
from sknetwork.clustering import PropagationClustering
from hdbscan import HDBSCAN
from umap import UMAP
import numpy as np
import pandas as pd
from sklearn.metrics import pair_confusion_matrix, silhouette_score, fowlkes_mallows_score, adjusted_mutual_info_score, adjusted_rand_score

import pandas as pd
import tqdm
from sknetwork.topology import get_connected_components





def ALGP_clustering_BM(sim, dmat): 
    np.fill_diagonal(dmat, 0)
    np.fill_diagonal(sim, 0)
    if np.all(sim == 0):
        return np.arange(len(sim))
    seuil=np.unique(sim)
    if len(seuil) > 1:  #sécurité matrice à 1
        seuil = seuil[:-1]
    partition = [PropagationClustering().fit_predict(sparse.csr_matrix(sim > th)) for th in seuil  ]
    sil = [silhouette_score(dmat, p, metric="precomputed") if (len(set(p)) > 1 and len(set(p))<len(sim)) else 0 for p in partition]
    return partition[np.argmax(np.array(sil))]



def ALGP_clustering_percentile_BM(sim, dmat): 
    np.fill_diagonal(dmat, 0)
    np.fill_diagonal(sim, 0)
    pourcentages = np.arange(2, 100, 1) 
    seuil = np.percentile(sim, pourcentages)
    partition = [PropagationClustering().fit_predict(sparse.csr_matrix(sim > th)) for th in seuil  ]
    sil = [silhouette_score(dmat, p, metric="precomputed") if (len(set(p)) > 1 and len(set(p))<len(sim)) else 0 for p in partition]
    return partition[np.argmax(np.array(sil))]



def ConnectedComponents_BM(sim, dmat):
    np.fill_diagonal(dmat, 0)
    np.fill_diagonal(sim, 0)
    if np.all(sim == 0):
        return np.arange(len(sim))
    seuil=np.unique(sim)
    if len(seuil) > 1:
        seuil = seuil[:-1]
    partitions = [get_connected_components(sparse.csc_matrix(sim >= t)) for t in seuil ]
    sil = [silhouette_score(dmat, p, metric="precomputed") if (len(set(p)) > 1 and len(set(p))<len(sim)) else 0 for p in partitions]
    return partitions[np.argmax(np.array(sil))]



def dissim_hdbscan_BM(dmat):
    """
    Clustering upon dissimilarity
    """
    np.fill_diagonal(dmat, 0)

    predictor = HDBSCAN(min_cluster_size=2,min_samples=1, metric="precomputed",match_reference_implementation=True)
    raw_hdbscan =  predictor.fit_predict(dmat)
    out_hdbscan = []
    
    # Handle that hdbscan.HDBSCAN does not produce clusters of size 2 (?)
    outout=dmat[:][raw_hdbscan==-1][:,raw_hdbscan==-1]
    
    outliers_indices = np.where(np.array(raw_hdbscan) == -1)[0]
    if len(outout) > 2:
        outliers_indices = np.where(np.array(raw_hdbscan) == -1)[0]
        closest = pd.DataFrame(outout).apply(lambda x: np.argpartition(x, 2)[1])
        tot = 0
        for idx,c in enumerate(closest):
            if closest[c] == idx and idx < c: 
                tot+=1 #, print(idx, c)
                clust_idx = idx + 1 + len(raw_hdbscan)
                raw_hdbscan[outliers_indices[ c ]] = clust_idx
                raw_hdbscan[outliers_indices[idx]] = clust_idx

    for i, l in enumerate(raw_hdbscan):
        if l == -1:
            out_hdbscan.append(i + len(raw_hdbscan))
        else:
            out_hdbscan.append(l)
    return out_hdbscan


def proj_hdbscan_BM(sim, dmat):
    """
    Clustering upon UMAP Projections    
    """

    np.fill_diagonal(dmat, 0)

    def partition_from_umap_dim(dim):
        mat = UMAP(n_components=dim, metric='precomputed', n_neighbors=15).fit_transform(dmat)
        predictor = HDBSCAN(min_cluster_size=2,min_samples=1 ,match_reference_implementation=True)
        raw_hdbscan =  predictor.fit_predict(mat)
        out_hdbscan = []
        
        # Handle that hdbscan.HDBSCAN does not produce clusters of size 2 (?)
        outout=dmat[:][raw_hdbscan==-1][:,raw_hdbscan==-1]
        if len(outout) > 2:
            outliers_indices = np.where(np.array(raw_hdbscan) == -1)[0]
            closest = pd.DataFrame(outout).apply(lambda x: np.argpartition(x, 2)[1])
            tot = 0
            for idx,c in enumerate(closest):
                if closest[c] == idx and idx < c: 
                    tot+=1 #, print(idx, c)
                    clust_idx = idx + 1 + len(raw_hdbscan)
                    raw_hdbscan[outliers_indices[ c ]] = clust_idx
                    raw_hdbscan[outliers_indices[idx]] = clust_idx

        for i, l in enumerate(raw_hdbscan):
            if l == -1:
                out_hdbscan.append(i + len(raw_hdbscan))
            else:
                out_hdbscan.append(l)
        return out_hdbscan
    
    partitions = [partition_from_umap_dim(d) for d in tqdm.tqdm(range(2, np.min([100, len(dmat)-1])),desc='Computing UMAP projections')]
    sil = [silhouette_score(dmat, p, metric="precomputed") if (len(set(p)) > 1 and len(set(p))<len(sim)) else 0 for p in tqdm.tqdm(partitions, desc='Computing Silhouettes')]
    return partitions[np.argmax(np.array(sil))]