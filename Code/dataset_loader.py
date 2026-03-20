import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Dictionnaire de correspondance entre les dossiers et les labels numériques
INSTRUMENTS = {'gac': 0, 'org': 1, 'pia': 2, 'voi': 3} 

class MusicDataset(Dataset):
    """
    Classe gérant le chargement des données. Elle parcourt les dossiers d'instruments,
    référence tous les fichiers de spectrogrammes (.pt) et permet à PyTorch d'y
    accéder via un index.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = []
        self.labels = []
        
        for inst_name, label in INSTRUMENTS.items():
            folder_path = os.path.join(data_dir, inst_name)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.pt'):
                        self.file_list.append(os.path.join(folder_path, file))
                        self.labels.append(label)

    def __len__(self):
        """ Retourne le nombre total d'échantillons présents dans le dataset. """
        return len(self.file_list)

    def __getitem__(self, idx):
        """ 
        Charge en mémoire le spectrogramme correspondant à l'index donné 
        et retourne le tuple (tenseur, étiquette). 
        """
        spec = torch.load(self.file_list[idx])
        label = self.labels[idx]
        return spec, label

def get_dataloaders(data_dir, batch_size=32):
    """
    Prépare les flux de données pour l'entraînement. Cette fonction divise le 
    dataset en deux sous-ensembles (75% entraînement, 25% validation) et crée 
    des objets DataLoader pour gérer le mélange et le groupement par lots (batches).
    """
    full_dataset = MusicDataset(data_dir)
    
    train_size = int(0.75 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    try:
        train_loader, val_loader = get_dataloaders("../processed_data/train")
        print(f"Nombre de lots d'entraînement : {len(train_loader)}")
        print(f"Nombre de lots de validation : {len(val_loader)}")
    except Exception as e:
        print(f"Erreur : Vérifiez que le chemin vers 'processed_data/train' est correct.")