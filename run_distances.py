from utils import get_config, get_images
from benchmark import xfeat_bm
from benchmark import utils_bm    


if __name__ == '__main__':

    #### Get Config
    cfg = get_config()

    #### Extract Similarities
    ds = cfg['Dataset']
    treasure_name = cfg['Treasure']
    
    images = utils_bm.get_images_in_treasure_dataset(treasure_name, ds)
    filtering = cfg['Filtering']
    OUTDIR = f"similarities/{treasure_name}/{ds}"
    FNAME = f"{OUTDIR}/matches.npy"
    DIST = f"{OUTDIR}/distances.npy"
    PATH_POI_COUPLES = f"{OUTDIR}/poi_couples" 


    if cfg['Matching']['Algo'] == "XFeat":
        xfeat_bm.build_similarity_matrix_from_poi_couples(images, filtering=filtering, dir_poi=PATH_POI_COUPLES, fname=FNAME)              
        xfeat_bm.create_distance_matrix(path_matrix_sim=FNAME, path_matrix_dist=DIST)
    elif cfg['Matching']['Algo'] == "RoMa":
        compute_roma_sim.save_matches(images, threshold=float(cfg['Matching']['Params']['RoMa-Threshold']), fname=FNAME)
    else:
        raise ValueError('Wrong matching algorithm selected. Must be in : XFeat | RoMa')
    



    #### Clustering
    #sim = np.load(FNAME)
    #partition = []
    #if cfg['Clustering'] == 'AGLP':
    #    partition = AGLP_clustering(sim)
    #elif cfg['Clustering'] == 'HDBSCAN-Proj':
    #    partition = proj_hdbscan(sim)
    #elif cfg['Clustering'] == 'HDBSCAN-Dissim':
    #    partition = dissim_hdbscan(sim)
    #else:
    #    raise ValueError('Wrong Clustering selected. Must be in : AGLP | HDBSCAN-Dissim | HDBSCAN-Proj ')

    #print(partition, partition.dtype, partition)
    #np.savetxt("die_studie.txt", partition, fmt="%i")
