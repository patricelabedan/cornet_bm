from pathlib import Path
from utils_bm import save_distance_matrix_to_csv, sort_csv_by_distance

path_ds = r"D:\PROJETS\COINS\ACCADIL\SRC\accadil-benchmark\accadil-benchmark\Datasets\SIMILARITIES\IJ_DS8\DS8"
path_mat_distance = str(Path(path_ds, "distances.npy"))
path_list = str(Path(path_ds, "files_list.txt"))
path_result_distances_not_sorted = str(Path(path_ds, "results_distance_not_sorted.csv"))
path_result_distances = str(Path(path_ds, "results_distance.csv"))

save_distance_matrix_to_csv(path_mat_distance, path_list, path_result_distances_not_sorted)

sort_csv_by_distance(path_result_distances_not_sorted, path_result_distances)



