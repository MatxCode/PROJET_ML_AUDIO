import os
import torch
import librosa
import numpy as np
from model import InstrumentCNN

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../models/best_model.pth"
TEST_DATA_PATH = "../Dataset/test"
SAMPLE_RATE = 22050
N_MELS = 128

INV_INSTRUMENTS = {0: 'gac', 1: 'org', 2: 'pia', 3: 'voi'}

def evaluate():
    """
    Charge le meilleur modèle sauvegardé et évalue ses performances sur le dataset de test.
    La fonction traite chaque fichier audio (.wav), génère son spectrogramme de Mel,
    et compare la prédiction du modèle avec les instruments réels listés dans le 
    fichier texte (.txt) correspondant. Calcule enfin le score de précision global.
    """
    model = InstrumentCNN(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    total_instances = 0
    correct_detections = 0

    print(f"Évaluation sur le dataset de test...")

    # parcours des fichiers de test
    files = [f for f in os.listdir(TEST_DATA_PATH) if f.endswith('.wav')]
    
    for wav_file in files:
        wav_path = os.path.join(TEST_DATA_PATH, wav_file)
        txt_path = wav_path.replace('.wav', '.txt')

        if not os.path.exists(txt_path):
            continue

        # taille variable
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # conversion en tenseur
        input_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # prédictions
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_instrument = INV_INSTRUMENTS[predicted_idx.item()]

        with open(txt_path, 'r') as f:
            content = f.read()
            true_instruments = content.split() 
        
        total_instances += 1
        
        # debug
        if total_instances <= 5:
            print(f"Fichier: {wav_file}")
            print(f"  > Prédit: {predicted_instrument}")
            print(f"  > Présents dans le fichier: {true_instruments}")

        if predicted_instrument in true_instruments:
            correct_detections += 1

    final_score = (correct_detections / total_instances) * 100
    print("-" * 30)
    print(f"Nombre de morceaux testés : {total_instances}")
    print(f"Score final (Métrique Projet) : {final_score:.2f}%")
    
    if final_score >= 70:
        print("Félicitations ! Objectif de 70% atteint.")
    else:
        print("Objectif non atteint, vérifiez l'équilibrage des données.")

if __name__ == "__main__":
    evaluate()