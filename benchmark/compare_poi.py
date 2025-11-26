import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- Paramètres ---
img_name = "5713-R"
# img_name = "12258-R"
# path_test1 = Path(r"C:\ACCADIL\MULTI_DATASETS\IJ_DS8_accadil\DS8\TEMP\KP AND DESC")
path_test1 = Path(r"C:\ACCADIL\MULTI_DATASETS\IJ_DS8_Light\DS8_Light\TEMP\KP AND DESC")
path_test2 = Path(r"D:\PROJETS\COINS\ACCADIL\SRC\accadil-benchmark\accadil-benchmark\Datasets\PREPROC\IJ_DS8_Light\DS8_Light\TEMP\KP AND DESC")

# --- Chargement des keypoints ---
def load_kp(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        kp = json.load(f)
    # kp doit être une liste de dicts ou de listes [x, y] ou [x, y, ...]
    return np.array(kp)

kp1 = load_kp(path_test1 / f"{img_name}.json")
kp2 = load_kp(path_test2 / f"{img_name}.json")

# --- Chargement des descripteurs ---
desc1 = np.load(path_test1 / f"{img_name}.npy")
desc2 = np.load(path_test2 / f"{img_name}.npy")

# --- Comparaison des keypoints ---
print(f"Nb keypoints test1: {len(kp1)}, test2: {len(kp2)}")
if np.array_equal(kp1, kp2):
    print("Keypoints identiques.")
else:
    print("Keypoints différents.")
    # Afficher les différences
    min_len = min(len(kp1), len(kp2))
    diff = np.abs(kp1[:min_len] - kp2[:min_len])
    print(f"Différence moyenne sur {min_len} premiers points: {diff.mean()}")

    # Affichage graphique
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.scatter(kp1[:,0], kp1[:,1], s=5, label="test1")
    plt.title("Keypoints test1")
    plt.subplot(1,2,2)
    plt.scatter(kp2[:,0], kp2[:,1], s=5, label="test2", color='orange')
    plt.title("Keypoints test2")
    plt.show()

# --- Comparaison des descripteurs ---
print(f"Descripteurs shape test1: {desc1.shape}, test2: {desc2.shape}")
if np.array_equal(desc1, desc2):
    print("Descripteurs identiques.")
else:
    print("Descripteurs différents.")
    min_len = min(len(desc1), len(desc2))
    diff = np.abs(desc1[:min_len] - desc2[:min_len])
    print(f"Différence moyenne sur {min_len} premiers descripteurs: {diff.mean()}")

    # Affichage graphique (optionnel, pour 2D ou 3D)
    if desc1.shape[1] == 2:
        plt.figure()
        plt.scatter(desc1[:min_len,0], desc1[:min_len,1], s=5, label="test1")
        plt.scatter(desc2[:min_len,0], desc2[:min_len,1], s=5, label="test2", alpha=0.5)
        plt.legend()
        plt.title("Descripteurs (2D)")
        plt.show()