import torch.nn as nn
import torch.nn.functional as F

TRAIN_HPAMS = {
        'None':      (55, 5e-4, 0.5),
        'ood_0.0_3': (3000, 5e-4, 0.2),
        'ood_0.0_5': (4000, 5e-4, 0.2),
        'ood_0.9_0': (2000, 5e-5, 1.0),
        'ood_1.0_0': (2000, 5e-5, 1.0)
    }

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        n_history = 5
        n_horizon = 5
        
        width = 200
        self.fc1 = nn.Linear(n_history*3, width)
        self.fc2 = nn.Linear(width, width)
        self.out = nn.Linear(width, n_horizon*2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x