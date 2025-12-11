from turtle import distance
from scipy import sparse
from sknetwork.clustering import PropagationClustering
from hdbscan import HDBSCAN
from umap import UMAP
import numpy as np
import pandas as pd
from sklearn.metrics import pair_confusion_matrix, silhouette_score, fowlkes_mallows_score, adjusted_mutual_info_score, adjusted_rand_score
from utils import get_labels
import pandas as pd
import tqdm
from sknetwork.topology import get_connected_components



def AGLP_clustering_OLD(sim, dmat):
    """
    Compute Graph from matches between coins, Label Propagation clustering
    Best graph select upon threshold
    """
    np.fill_diagonal(sim, 0)
    sim = sim * 1000  #ASTUCE!!! car int(sim.max()) mettait tout à 0
    partitions = [PropagationClustering().fit_predict(sparse.csr_matrix(sim > th)) for th in tqdm.tqdm(range(int(sim.max())), desc="Computing partitions for each threshold")]
    sil = [silhouette_score(dmat, p, metric="precomputed") if (len(set(p)) > 1 and len(set(p))<len(sim)) else 0 for p in tqdm.tqdm(partitions, desc='Computing Silhouettes')]
    print(sil)
    print(f'Optimal threshold : {np.argmax(sil)}')
    return partitions[np.argmax(np.array(sil))]



def AGLP_clustering_2(sim, dmat):

    print("clustering... AGLP_2...")

    np.fill_diagonal(dmat, 0) #TODO remove and put '0'
    print(dmat.max())
    sim = dmat.max() - dmat
    np.fill_diagonal(sim, 0)
    vals = np.unique(sim)
    # partitions = [PropagationClustering().fit_predict(sparse.csr_matrix(sim > th)) for th in tqdm.tqdm(vals[:-1], desc="Computing partitions for each threshold")]
    partitions = []
    for th in tqdm.tqdm(vals[:-1], desc="Computing partitions for each threshold"):
        # print(f"Current threshold: {th}")
        partition = PropagationClustering().fit_predict(sparse.csr_matrix(sim > th))
        partitions.append(partition)                
    sil = [silhouette_score(dmat, p, metric="precomputed") if (len(set(p)) > 1 and len(set(p))<len(sim)) else 0 for p in tqdm.tqdm(partitions, desc='Computing Silhouettes')]
    print('sil', sil)
    print(f'Optimal threshold : {np.argmax(sil)}')
    cmap_pred = partitions[np.argmax(np.array(sil))]
    print('cmap_pred', cmap_pred)
                
    print("clustering... AGLP_2... OK")
    return cmap_pred


def ConnectedComponents_clustering_2(sim, dmat):
    np.fill_diagonal(sim, 0)
    partitions = [get_connected_components(sparse.csc_matrix(sim > t)) for t in tqdm.tqdm(range(int(sim.max())), desc="Computing connected components")]
    sil = [silhouette_score(dmat, p, metric="precomputed") if (len(set(p)) > 1 and len(set(p))<len(sim)) else 0 for p in tqdm.tqdm(partitions, desc='Computing Silhouettes')]
    print(f'Optimal threshold : {np.argmax(sil)}')
    return partitions[np.argmax(np.array(sil))]


def dissim_hdbscan_2(dmat):
    """
    Clustering upon dissimilarity
    """
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


def proj_hdbscan_2(sim, dmat):
    """
    Clustering upon UMAP Projections    
    """
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



def evaluate_clustering(partition, labels_true):
    pmat = pair_confusion_matrix(partition, labels_true)
    return {
        'AMI' : adjusted_mutual_info_score(labels_true, partition),
        'ARI' : adjusted_rand_score(labels_true, partition),
        'FMI' : fowlkes_mallows_score(labels_true, partition),
        'Pairwise Precision' : pmat[1][1] / (pmat[0][1] + pmat[1][1]),
        'Pairwise Recall' : pmat[1][1] / (pmat[1][0] + pmat[1][1])
    }


if __name__ == '__main__':
    sim = np.load('similarities_dissimilarities/xfeat_matches.npy')
    print('AGLP 2')
    AGLP_clustering_2(sim)
    print('HDBSCAN dissim 2')
    dissim_hdbscan_2(sim)
    print('HDBSCAN Proj 2')
    proj_hdbscan_2(sim)