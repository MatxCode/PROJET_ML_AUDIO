import torch
import torch.nn as nn
import torch.optim as optim
from model import InstrumentCNN
from dataset_loader import get_dataloaders
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30 
LEARNING_RATE = 0.001
BATCH_SIZE = 32

def train():
    """
    Lance la procédure complète d'entraînement du modèle.
    
    Cette fonction :
    1. Prépare les dossiers et charge les données via les DataLoaders.
    2. Initialise le modèle InstrumentCNN, la fonction de coût (CrossEntropy) et l'optimiseur (Adam).
    3. Exécute la boucle d'entraînement sur plusieurs époques.
    4. Évalue la précision sur le jeu de validation après chaque époque.
    5. Sauvegarde automatiquement l'état du modèle ayant obtenu la meilleure précision.
    """
    
    # si dossier models n'existe pas
    if not os.path.exists("../models"):
        os.makedirs("../models")

    # chargement des données
    train_loader, val_loader = get_dataloaders("../processed_data/train", batch_size=BATCH_SIZE)
    
    model = InstrumentCNN(num_classes=4).to(DEVICE)
    
    # optimiseur & loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Démarrage de l'entraînement sur : {DEVICE}")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train() # active dropout et batchnorm
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        model.eval() # désactive dropout
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Val Acc: {val_accuracy:.2f}%")

        # sauvegarde du meilleur modèle
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "../models/best_model.pth")
            print(f"--> Modèle sauvegardé (Acc: {val_accuracy:.2f}%)")

    print(f"\nEntraînement terminé ! Meilleure précision en validation : {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()