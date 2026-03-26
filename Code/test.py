import gradio as gr
import torch
import librosa
import numpy as np
from model import InstrumentCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../models/best_model.pth"
INV_INSTRUMENTS = {0: 'Guitare Acoustique (gac)', 1: 'Orgue (org)', 
                  2: 'Piano (pia)', 3: 'Voix Humaine (voi)'}

# chargement du modèle
model = InstrumentCNN(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict_instrument(audio_path):
    if audio_path is None:
        return "Veuillez charger un fichier audio."
    
    # .load est par défault en mono, donc pas de mono dans l'appel de fonction 
    y, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    input_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # prédiction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # dictionnaire d'instruments
    results = {INV_INSTRUMENTS[i]: float(probabilities[i]) for i in range(4)}
    return results

# config interface web
demo = gr.Interface(
    fn=predict_instrument,
    inputs=gr.Audio(type="filepath", label="Déposez votre morceau ici"),
    outputs=gr.Label(num_top_classes=4, label="Prédictions"),
    title="🎸 Reconnaissance d'instruments",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()