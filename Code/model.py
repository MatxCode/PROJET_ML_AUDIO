import torch
import torch.nn as nn
import torch.nn.functional as F

class InstrumentCNN(nn.Module):
    """
    Réseau de neurones convolutif (CNN) conçu pour la classification d'instruments de musique.
    
    L'architecture utilise trois blocs de convolutions suivis d'un pooling adaptatif.
    Cette structure permet au modèle de gérer des spectrogrammes de durées variables
    tout en retournant une sortie de taille fixe pour la classification finale.
    """
    def __init__(self, num_classes=4):
        """
        Définit les couches du modèle : 3 couches de convolution avec Batch Normalization,
        une couche de pooling adaptatif, et deux couches linéaires pour la décision.
        """
        super(InstrumentCNN, self).__init__()
        
        # convolution 1
        self.bloc1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # convolution 2
        self.bloc2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # convolution 3
        self.bloc3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # pooling adaptatif
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5), # sur-apprentissage
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Définit le flux de données à travers le réseau (Passage vers l'avant).
        Applique successivement les convolutions, les activations ReLU, le Max Pooling,
        puis finit par les couches de classification.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        
        x = self.adaptive_pool(x) 
        
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = InstrumentCNN()
    test_input = torch.randn(1, 1, 128, 130)
    output = model(test_input)
    print(f"Sortie pour 3s : {output.shape}")
    
    test_input_long = torch.randn(1, 1, 128, 500)
    output_long = model(test_input_long)
    print(f"Sortie pour taille variable : {output_long.shape}") 