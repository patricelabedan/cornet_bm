from utils import get_config, get_images
from benchmark import xfeat_bm, utils_bm, roma_bm    
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

    #### Init paths or vars
    images = utils_bm.get_images_in_treasure_dataset(treasure_name, ds)
    OUTDIR = f"similarities/{treasure_name}/{ds}"
    FNAME = f"{OUTDIR}/matches.npy"
    DIST = f"{OUTDIR}/distances.npy"
    PATH_POI_COUPLES = f"{OUTDIR}/poi_couples" 
    FILES_LIST = f"{OUTDIR}/files_list.txt"

    #### poi-couples -> similarity 
    if cfg['Distance'] == "XFeat":  # CADS, ...?
        xfeat_bm.build_similarity_matrix_from_poi_couples(images, filtering=filtering, dir_poi=PATH_POI_COUPLES, fname=FNAME)              
    elif cfg['Distance'] == "RoMa":
        roma_bm.build_similarity_matrix_from_poi_couples(images, filtering=filtering, dir_poi=PATH_POI_COUPLES, fname=FNAME)
    else:
        raise ValueError('Wrong matching algorithm selected. Must be in : XFeat | RoMa')
    
    #### similarity -> distance 
    utils_bm.create_distance_matrix(path_matrix_sim=FNAME, path_matrix_dist=DIST, path_coin_list_txt=FILES_LIST)