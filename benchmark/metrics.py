# from pathlib import Path
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, recall_score, precision_score, f1_score, accuracy_score
# import json




# def loadGroundTruthDS(path_treasure, ds):
#     '''
#     Load GroundTruth.csv for a given dataset and treasure, and return the list of links (pairs of coins that are linked).
#     '''

#     _function = "_loadGroundTruthDS_"
#     _dump = True

#     if _dump:
#         print(f"{_function} Loading GroundTruth.csv in ### {path_treasure} ###...")
#         print(f"{_function} path_treasure : {path_treasure}")
#         print(f"{_function} ds : {ds}")

#     path_ground_truth = str(Path(path_treasure, 'GroundTruth.csv'))
#     if _dump:
#         print(f"{_function} path_ground_truth : {path_ground_truth}")

#     ground_truths = pd.read_csv(str(path_ground_truth), sep=";", header=None)
#     if _dump:
#         print(f"{_function} ground_truths : {ground_truths}")

#     ground_truth = ground_truths.loc[ground_truths[0] == ds]
#     if _dump:
#         print(f"{_function} ground_truth : {ground_truth}")

#     links = ground_truth[[1,2]].values.tolist()
#     if _dump:
#         print(f"{_function} links : {links}")        

#     return links




# def defineGroundTruthMatrix(ref_pic_list, links):
#     '''
#     Define the GroundTruth matrix for a given list of reference pictures and links.
#     '''

#     _function = "_defineGroundTruthMatrix_"
#     _dump = True

#     nb = len(ref_pic_list)
#     if _dump:
#         print(f"{_function} nb coins : {nb}")

#     ref_pic_stems = [Path(p).stem for p in ref_pic_list]

#     G = np.full((nb,nb), False)
#     for l in links:
#         c1,c2 = ref_pic_stems.index(Path(l[0]).stem),ref_pic_stems.index(Path(l[1]).stem)
#         G[min(c1,c2),max(c1,c2)] = True
    
#     if _dump:
#         print(f"{_function} G : {G}")


#     return G




# def histogram(axs, treasure_name, ds_name, nbCoins, distance, G, PATH_PROJECT, path_supp=""):
#     '''
#     Draw histograms for the given dataset and treasure.
#     '''
#     _function = "_histogram_"
#     _dump = True

#     UT = np.full((nbCoins,nbCoins), False)
#     UT[np.triu_indices(nbCoins,k=1)] = True
#     NG = np.logical_and(np.logical_not(G),UT)

#     distances = np.concatenate((distance[NG],distance[G]))
#     if _dump:
#         print(f"{_function} distances : {distances}")

#     link =  np.concatenate((['no']*len(distance[NG]),['yes']*len(distance[G])))
#     if _dump:
#         print(f"{_function} link : {link}")

#     df = pd.DataFrame({'distance': distances, 'link': link })
#     # fig, axs = plt.subplots(figsize=(8,2))
#     #plt.tight_layout(pad=0.5,w_pad=2.5,rect=(0,0,0.96,1))
#     bins = np.linspace(df['distance'].min(),df['distance'].max(),50) 
#     #bins = np.linspace(0.3,0.5,50)
#     sns.histplot(data=df[df['link']=='yes'], x='distance', ax=axs, color="red", alpha=0.4, bins = bins)
#     ax2 = axs.twinx()
#     sns.histplot(data=df[df['link']=='no'], x='distance', ax=ax2, kde=True, hue='link', bins = bins)
#     ax2.set(ylabel=None), axs.set(ylabel=None), ax2.set(xlabel=None), axs.set(xlabel=None)
#     plt.title(ds_name)
#     # plt.savefig(str(Path(PATH_PROJECT, 'Datasets','SIMILARITIES', treasure_name, ds)) + '.png')
#     # plt.savefig(str(Path(PATH_PROJECT, 'Datasets','SIMILARITIES', treasure_name, ds_name, f"histogram_{ds_name}")) + '.png')
#     plt.savefig(str(Path(PATH_PROJECT, 'Datasets','SIMILARITIES', treasure_name, ds_name, f"histogram_{ds_name}_{path_supp}")) + '.jpg') 





def compute_y_true(nbCoins, G):
    '''
    Compute the true links (y_true) from the GroundTruth matrix G.
    '''
    y_true = G[np.triu_indices(nbCoins,k=1)]
    return y_true




def computing_metrics(nbCoins, G, cmap_pred_path):
    '''
    Compute performance metrics for the given dataset and treasure.
    '''
    _function = "_computing_metrics_"
    _dump = True

    print(f"[{_function}] Starting computation of metrics...")
    print(f"[{_function}] nbCoins : {nbCoins}, G : {G}")
    print(f"[{_function}] cmap_pred_path : {cmap_pred_path}")

    
    ## cmap_pred (reload)
    cmap_pred = np.loadtxt(cmap_pred_path, dtype=int)
    if _dump:
        print(f"[{_function}] cmap_pred : {cmap_pred}")

    ## y_true
    y_true = compute_y_true(nbCoins, G)
    if _dump:
        print(f"[{_function}] y_true : {y_true}")

    # True cluster map (cmap_true maps elements to cluster indices)
    cmap_true = compute_true_cluster_map(nbCoins, G)
    if _dump:
        print(f"[{_function}] cmap_true : {cmap_true}")
            
    ## y_pred
    y_pred = compute_y_pred(nbCoins, cmap_pred)
    if _dump:
        print(f"[{_function}] y_pred : {y_pred}")

    ## Compute performance metrics
    ARI, NMI, precision, recall, f1score, accuracy = \
        compute_performance_metrics(cmap_true, cmap_pred, y_true, y_pred)

    return ARI, NMI, precision, recall, f1score, accuracy           



def compute_performance_metrics(cmap_true, cmap_pred, y_true, y_pred):
    '''
    Compute performance metrics (ARI, NMI, precision, recall, f1score, accuracy) 
    for the given true and predicted cluster maps and links.
    '''

    _function = "_compute_performance_metrics_"
    _dump = True

    ARI = adjusted_rand_score(cmap_true, cmap_pred)
    if _dump:
        print(f"{_function} Adjusted Rand Index : {ARI}")
    NMI = normalized_mutual_info_score(cmap_true, cmap_pred)
    if _dump:
        print(f"{_function} Normalized Mutual Information : {NMI}")
    precision = precision_score(y_true, y_pred)
    if _dump:
        print(f"{_function} Precision = 1-FDR : {precision}")
    recall = recall_score(y_true, y_pred)
    if _dump:
        print(f"{_function} Recall = Sensitivity : {recall}")
    f1score = f1_score(y_true, y_pred)
    if _dump:
        print(f"{_function} F1 score : {f1score}")
    accuracy = accuracy_score(y_true, y_pred)
    if _dump:
        print(f"{_function} Accuracy : {accuracy}")     

    return ARI, NMI, precision, recall, f1score, accuracy           




def compute_true_cluster_map(nbCoins, G):
    '''
    Compute the true cluster map (cmap_true) from the GroundTruth matrix G.
    The true cluster map (cmap_true) maps each coin to a cluster index, 
    where coins that are linked (i.e., have a True value in G) belong to the same cluster.
    '''
    _function = "_compute_true_cluster_map_"
    _dump = True
    _dump_while = True

    print(f"[{_function}] Starting computation of true cluster map...")
    print(f"[{_function}] nbCoins : {nbCoins}")
    print(f"[{_function}] G : {G}")

    indices_temp = np.arange(nbCoins)
    if _dump:
        print(f"[{_function}] indices_temp : {indices_temp}")

    cind, cmap_true = 0, np.array([-1]*nbCoins)
    if _dump:
        print(f"[{_function}] cmap_true : {cmap_true}")
        print(f"[{_function}] cind : {cind}")
        print(f"[{_function}] npargwhereG : {np.argwhere(G[indices_temp[0],:])}")

    while len(indices_temp)!=0:
        current = indices_temp[0]
        # G is upper-triangular only (G[i,j] set for i<j): look at both row and column
        # of "current", and always include "current" itself (it may have no links at all).
        linked_row = [index[0] for index in np.argwhere(G[current, :])]
        linked_col = [index[0] for index in np.argwhere(G[:, current])]
        indices = sorted(set([current] + linked_row + linked_col))
        if _dump_while:
            print(f"[{_function}] indices : {indices}")
        to_remove = [np.where(indices_temp == ind)[0] for ind in indices]
        if _dump_while:
            print(f"[{_function}] to_remove : {to_remove}")
        indices_temp = np.delete(indices_temp,to_remove)
        if _dump_while:
            print(f"[{_function}] indices_temp : {indices_temp}")
        cmap_true[indices] = cind
        if _dump_while:
            print(f"[{_function}] cmap_true : {cmap_true}")
            print(f"[{_function}] ---------")
        cind+=1

    return cmap_true



def compute_y_pred(nbCoins, cmap_pred):
    '''
    Compute the predicted links (y_pred) from the predicted cluster map (cmap_pred).
    '''
    P = np.full((nbCoins,nbCoins), 0)
    for c in cmap_pred:
        v = np.zeros((nbCoins,1))
        indices = np.where(cmap_pred == c)[0]
        for index in indices:
            v[index][0] = 1
        P = P + v @ v.transpose()
    P = P>0
    y_pred = P[np.triu_indices(nbCoins,k=1)]    
    return y_pred





# def saveBestThresholds(path_destination, best_thresholds):
#     '''
#     Save the best thresholds for each dataset and treasure in a JSON file.
#     ''' 
#     _function = "_saveBestThresholds_"
#     _dump = True

#     if _dump:
#         print(f"{_function} Saving best thresholds... ")

#     if _dump:
#         print(f"{_function} path_destination : {path_destination}")
#         print(f"{_function} best_thresholds : {best_thresholds}")

#     Path(path_destination).mkdir(parents=True, exist_ok=True)
#     bthreshold_path = Path(path_destination, 'best_thresholds.json')
#     if _dump:
#         print(f"{_function} bthreshold_path : {bthreshold_path}")

#     with open(bthreshold_path, 'w') as f:
#         json.dump(best_thresholds, f, indent=4)    

#     if _dump:
#         print(f"{_function} Saving best thresholds... OK")





# def saveBestThresholds2D(path_destination, best_thresholds, algo_list):
#     '''
#     Save the best thresholds for each dataset and treasure in a JSON file.
#     ''' 
#     _function = "_saveBestThresholds2D_"
#     _dump = True

#     if _dump:
#         print(f"{_function} Saving best thresholds... ")

#     if _dump:
#         print(f"{_function} path_destination : {path_destination}")
#         print(f"{_function} best_thresholds : {best_thresholds}")
#         print(f"{_function} algo_list : {algo_list}")


#     for algo in algo_list:     

#         Path(path_destination).mkdir(parents=True, exist_ok=True)
#         bthreshold_path = Path(path_destination, f'best_thresholds_{algo}.json')
#         if _dump:
#             print(f"{_function} bthreshold_path : {bthreshold_path}")

#         with open(bthreshold_path, 'w') as f:
#             json.dump(best_thresholds[algo], f, indent=4)    



#     if _dump:
#         print(f"{_function} Saving best thresholds... OK")
