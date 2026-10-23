from utils import get_config, get_images
from benchmark import xfeat_bm, utils_bm, roma_bm    
from pathlib import Path
import os


if __name__ == '__main__':

    _dump = True
    _function = "[cornet_bm] [run_distances.py]"

    #### Get Config
    cfg = get_config()

    #### Extract Similarities
    ds = cfg['Dataset']
    PATH_ABS_PROJET = cfg['PATH_PROJECT']
    treasure_name = cfg['Treasure']
    algo_matching = cfg['Matching']['Algo']
    filtering = cfg['Filtering']
    SIMILARITIES = cfg['SIMILARITIES']
    cfg_distance = cfg['Distance']
    path_images = cfg['path_images']

    #### Init paths or vars
    # images = utils_bm.getListImagesInFolder(str(Path(PATH_ABS_PROJET, "Datasets", "BASE", treasure_name, ds)))
    images = utils_bm.getListImagesInFolder(str(path_images))
    if _dump:
        print(f"{_function} images = {images}")

    ### IN
    path_similarities_ds_cornet = str(Path("similarities", treasure_name, ds))
    path_similarities_ds_accadil = str(Path(PATH_ABS_PROJET, "Datasets", "SIMILARITIES", treasure_name, ds))
    
    if algo_matching.lower() in ['xfeat', 'roma']:
        INDIR = path_similarities_ds_cornet
    if algo_matching.lower() in ['bestsift']:
        INDIR = path_similarities_ds_accadil
    PATH_POI_COUPLES = str(Path(INDIR, "poi_couples"))

    ### clean similarities cornet files if matching accadil
    if algo_matching.lower() in ['bestsift']:
        path_poi_couples_cornet = str(Path(path_similarities_ds_cornet, "poi_couples"))
        utils_bm.clean_poi_couples_directory(out_dir=path_poi_couples_cornet)


    ### OUT
    OUTDIR = Path("similarities", treasure_name, ds)
    os.makedirs(OUTDIR, exist_ok=True)
    OUTDIR = str(OUTDIR)

    FNAME = str(Path(OUTDIR, "matches.npy"))
    DIST = str(Path(OUTDIR, "distances.npy"))
    FILES_LIST = str(Path(INDIR, "files_list.txt"))
    # FILES_LIST = str(Path(OUTDIR, "files_list.txt"))

    MATCHES_FILES = str(Path(OUTDIR, "matches_files.txt"))
    RESULTS_DISTANCE = str(Path(OUTDIR, "results_distance.csv"))
    RESULTS_DISTANCE_NOT_SORTED = str(Path(OUTDIR, "results_distance_not_sorted.csv"))

    if _dump:
        print(f"{_function} INDIR = {INDIR}")
        print(f"{_function} OUTDIR = {OUTDIR}")
        print(f"{_function} FNAME = {FNAME}")
        print(f"{_function} DIST = {DIST}")
        print(f"{_function} PATH_POI_COUPLES = {PATH_POI_COUPLES}")
        print(f"{_function} FILES_LIST = {FILES_LIST}")
        print(f"{_function} MATCHES_FILES = {MATCHES_FILES}")
        print(f"{_function} RESULTS_DISTANCE = {RESULTS_DISTANCE}")
        print(f"{_function} RESULTS_DISTANCE_NOT_SORTED = {RESULTS_DISTANCE_NOT_SORTED}")


    ### clean outputs
    if os.path.exists(FNAME):
        os.remove(FNAME)
    if os.path.exists(DIST):
        os.remove(DIST)
    # if os.path.exists(FILES_LIST):
    #     os.remove(FILES_LIST)
    if os.path.exists(MATCHES_FILES):
        os.remove(MATCHES_FILES)
    if os.path.exists(RESULTS_DISTANCE):
        os.remove(RESULTS_DISTANCE)
    if os.path.exists(RESULTS_DISTANCE_NOT_SORTED):
        os.remove(RESULTS_DISTANCE_NOT_SORTED)

    # exit(1)

    #### poi-couples -> similarity 
    if cfg['Distance'] == "XFeat":  # CADS, ...?
        if algo_matching.lower() in ['bestsift']:
            xfeat_bm.build_similarity_matrix_from_poi_couples_accadil(images, filtering=filtering, dir_poi=PATH_POI_COUPLES, fname=FNAME) 
        else:
            xfeat_bm.build_similarity_matrix_from_poi_couples(images, filtering=filtering, dir_poi=PATH_POI_COUPLES, fname=FNAME)              
    elif cfg['Distance'] == "RoMa":
        roma_bm.build_similarity_matrix_from_poi_couples(images, filtering=filtering, dir_poi=PATH_POI_COUPLES, fname=FNAME)
    else:
        raise ValueError('Wrong matching algorithm selected. Must be in : XFeat | RoMa')
    

    
    #### similarity -> distance 
    utils_bm.create_distance_matrix(path_matrix_sim=FNAME, path_matrix_dist=DIST, path_coin_list_txt=FILES_LIST)