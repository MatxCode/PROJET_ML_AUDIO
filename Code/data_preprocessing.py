import os
import torch
import librosa
import numpy as np
from tqdm import tqdm

# config
DATASET_PATH = "../Dataset/Train_gac/gac"
OUTPUT_PATH = "../processed_data/train"
SAMPLE_RATE = 22050
N_MELS = 128

def save_spectrograms(src_path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for root, dirs, files in os.walk(src_path):
        for file in tqdm(files):
            if file.endswith('.wav'):
                # chargement du son
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # transformation en spectrogramme
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # sauvegade au format pytorch
                category = os.path.basename(root)
                category_dir = os.path.join(dest_path, category)
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)
                
                output_file = os.path.join(category_dir, file.replace('.wav', '.pt'))
                torch.save(torch.FloatTensor(mel_spec_db), output_file)

if __name__ == "__main__":
    print("Début de la transformation des fichiers d'entraînement...")
    save_spectrograms(DATASET_PATH, OUTPUT_PATH)