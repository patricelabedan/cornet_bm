
import glob
import numpy as np
import csv
import shutil
import os


def get_images_in_treasure_dataset(treasure, dataset):

    files = glob.glob(f"datasets/{treasure}/{dataset}/*.*")
    files = list(
        filter(
            lambda f: f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.gif') or f.endswith('.tiff'), 
            files
            )
        )
    return sorted(files)






