import torch.nn as nn
import torch.nn.functional as F


class SimpleSpeechModel(nn.Module):
    def __init__(self, n_classes):
        super(SimpleSpeechModel, self).__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128, n_classes)

    def forvard(self, x):
        mel = x['mel_spec'].unsqueeze(0)
        mel = mel.unsqueeze(0) if mel.dim() == 3 else mel

        z = F.relu(self.conv1(mel))
        z = F.relu(self.conv2(z))
        z = self.pool(z)
        z = z.view(z.size(0), -1)
        z = F.relu(self.fc1(z))
        out = self.fc2(z)
        return out