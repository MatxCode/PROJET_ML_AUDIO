import torch
import torch.nn as nn
import torch.nn.functional as F

class InstrumentCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(InstrumentCNN, self).__init__()
        
        # convolution 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # convolution 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # convolution 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # polling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5) # surapprentissage
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = F.relu(self.bn3(F.max_pool2d(self.conv3(x), 2)))
        
        x = self.adaptive_pool(x) 
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = InstrumentCNN()
    test_input = torch.randn(1, 1, 128, 130)
    output = model(test_input)
    print(f"Sortie pour 3s : {output.shape}")
    
    test_input_long = torch.randn(1, 1, 128, 500)
    output_long = model(test_input_long)
    print(f"Sortie pour taille variable : {output_long.shape}") 