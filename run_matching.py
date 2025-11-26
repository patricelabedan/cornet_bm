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
    OUTDIR = f"similarities/{treasure_name}/{ds}/"
    FNAME = f"{OUTDIR}/matches.npy"
    DIST = f"{OUTDIR}/distances.npy"
    PATH_POI_COUPLES = f"{OUTDIR}/poi_couples" 


    if cfg['Matching']['Algo'] == "XFeat":
        xfeat_bm.clean_poi_couples_directory(out_dir=PATH_POI_COUPLES)
        xfeat_bm.save_poi_couples(images, top_k=int(cfg['Matching']['Params']['XFeat-TopK']), out_dir=PATH_POI_COUPLES)
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
