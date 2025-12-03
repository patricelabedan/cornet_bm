from utils import get_config, get_images
from benchmark import xfeat_bm
from benchmark import utils_bm    
from pathlib import Path


if __name__ == '__main__':

    #### Get Config
    cfg = get_config()

    #### Extract Similarities
    ds = cfg['Dataset']
    treasure_name = cfg['Treasure']
    PATH_PROJET = cfg['PATH_PROJECT']
    SIMILARITIES = cfg['SIMILARITIES']
    filtering = cfg['Filtering']

    # Init paths or vars
    PATH_SIMILARITIES_DS = str(Path(PATH_PROJET, SIMILARITIES, treasure_name, ds))    
    images = utils_bm.get_images_in_treasure_dataset(treasure_name, ds)
    OUTDIR = f"similarities/{treasure_name}/{ds}"
    FNAME = f"{OUTDIR}/matches.npy"
    DIST = f"{OUTDIR}/distances.npy"
    PATH_POI_COUPLES = f"{OUTDIR}/poi_couples" 

    if cfg['Distance'] == "CORNET":  # xfeat ou Roma même chose? et CADS, ...?
    #     print ("Algo matching : " , cfg['Matching']['Algo'])
        xfeat_bm.build_similarity_matrix_from_poi_couples(images, filtering, PATH_POI_COUPLES, FNAME)              
        xfeat_bm.create_distance_matrix(FNAME, DIST, PATH_SIMILARITIES_DS)
    # elif cfg['Matching']['Algo'] == "RoMa":
    #     compute_roma_sim.save_matches(images, threshold=float(cfg['Matching']['Params']['RoMa-Threshold']), fname=FNAME)
    else:
        raise ValueError('Wrong matching algorithm selected. Must be in : XFeat | RoMa')
    
