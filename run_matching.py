from pathlib import Path
from utils import get_config, get_images
from benchmark import xfeat_bm, utils_bm, roma_bm  
import os


if __name__ == '__main__':

    #### Get Config
    cfg = get_config()

    #### Extract Similarities
    ds = cfg['Dataset']
    treasure_name = cfg['Treasure']
    algo_matching = cfg['Matching']['Algo']
    path_images = cfg['path_images']

    #### Paths or vars
    # images = utils_bm.get_images_in_treasure_dataset(treasure_name, ds)
    images_rel = utils_bm.getListImagesInFolder(path_images)
    images = [str(Path(path_images) / p) for p in images_rel]

    OUTDIR = Path("similarities", treasure_name, ds)
    os.makedirs(OUTDIR, exist_ok=True)
    OUTDIR = str(OUTDIR)

    PATH_POI_COUPLES = str(Path(OUTDIR, "poi_couples"))
    

    if algo_matching.lower() == "xfeat":
        utils_bm.clean_poi_couples_directory(out_dir=PATH_POI_COUPLES)
        xfeat_bm.save_poi_couples(images, top_k=int(cfg['Matching']['Params']['XFeat-TopK']), out_dir=PATH_POI_COUPLES)
    elif algo_matching.lower() == "roma":
        utils_bm.clean_poi_couples_directory(out_dir=PATH_POI_COUPLES)
        roma_bm.save_poi_couples(images, threshold=float(cfg['Matching']['Params']['RoMa-Threshold']), out_dir=PATH_POI_COUPLES)
    else:
        raise ValueError('Wrong matching algorithm selected. Must be in : XFeat | RoMa')
    

