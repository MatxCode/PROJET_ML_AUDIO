import gradio as gr
import torch
import librosa
import numpy as np
from model import InstrumentCNN

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../models/best_model.pth"
INV_INSTRUMENTS = {0: 'Guitare Acoustique (gac)', 1: 'Orgue (org)', 
                  2: 'Piano (pia)', 3: 'Voix Humaine (voi)'}

# Charger le modèle une seule fois
model = InstrumentCNN(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict_instrument(audio_path):
    if audio_path is None:
        return "Veuillez charger un fichier audio."
    
    # 1. Prétraitement identique au test [cite: 32]
    y, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 2. Conversion en tenseur
    input_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # 3. Prédiction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Créer un dictionnaire de résultats pour l'affichage
    results = {INV_INSTRUMENTS[i]: float(probabilities[i]) for i in range(4)}
    return results

# Configuration de l'interface Web
demo = gr.Interface(
    fn=predict_instrument,
    inputs=gr.Audio(type="filepath", label="Déposez votre morceau ici"),
    outputs=gr.Label(num_top_classes=4, label="Prédictions"),
    title="🎸 Reconnaissance d'Instruments - ESIEA ML Project",
    description="Projet INF3043 : Identifiez l'instrument dominant parmi la guitare, l'orgue, le piano ou la voix."
)

if __name__ == "__main__":
    demo.launch() # Lance un serveur web local (http://127.0.0.1:7860)