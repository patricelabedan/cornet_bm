
import glob
import numpy as np
import csv

def save_distance_matrix_to_csv(distance_matrix_path, ref_pic_list_path, output_csv_path):
    distances = np.load(distance_matrix_path)
    with open(ref_pic_list_path, mode='r', encoding='utf-8') as f:
        ref_pic_list = [line.strip() for line in f if line.strip()]
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['name1', 'name2', 'Distance'])
        n = len(ref_pic_list)
        for i in range(n):
            for j in range(i+1, n):
                writer.writerow([ref_pic_list[i], ref_pic_list[j], distances[i, j]])


def sort_csv_by_distance(input_csv_path, output_csv_path):
    with open(input_csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        rows = list(reader)
        rows.sort(key=lambda x: float(x['Distance']))

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(rows)

def get_images_in_treasure_dataset(treasure, dataset):

    files = glob.glob(f"datasets/{treasure}/{dataset}/*.*")
    files = list(
        filter(
            lambda f: f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.gif') or f.endswith('.tiff'), 
            files
            )
        )
    return sorted(files)



