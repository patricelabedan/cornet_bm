
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

def copy_folder_recursive_keep(keep_dest: bool, 
                               path_folder_src: str, 
                               path_folder_dst: str) -> bool:
    """
    Recursively copy the entire contents of path_folder_src to path_folder_dst.

    Parameters:
        keep_dest (bool): If True and path_folder_dst exists, do nothing. If False and path_folder_dst exists, delete it before copying.
        path_folder_src (str): Source folder path to copy from.
        path_folder_dst (str): Destination folder path to copy to.

    Returns:
        bool: True if the operation was successful, False otherwise.

    Raises:
        FileNotFoundError: If the source folder does not exist.
    """

    if not os.path.exists(path_folder_src):
        print(f"Error: Source folder does not exist: {path_folder_src}")
        return False

    if os.path.exists(path_folder_dst):
        if keep_dest:
            # Destination exists and we keep it
            return True
        else:
            try:
                shutil.rmtree(path_folder_dst)
            except Exception as e:
                print(f"Error removing destination folder: {e}")
                return False

    try:
        shutil.copytree(path_folder_src, path_folder_dst)
        return True
    except Exception as e:
        print(f"Error copying folder: {e}")
        return False



def copy_folder_recursive(path_folder_src: str, 
                          path_folder_dst: str) -> bool:
    """
    Recursively copy the entire contents of path_folder_src to path_folder_dst.

    If path_folder_dst exists, it will be deleted before copying.
    If path_folder_src does not exist, an error message is printed.

    Parameters:
        path_folder_src (str): Source folder path to copy from.
        path_folder_dst (str): Destination folder path to copy to.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    return copy_folder_recursive_keep(False, path_folder_src, path_folder_dst)




