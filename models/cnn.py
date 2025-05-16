import torch
import torch.nn as nn

class AntennaCNN(nn.Module):
    def __init__(self, grid_size=7):
        super(AntennaCNN, self).__init__()
        self.grid_size = grid_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        flat_size = 32 * (grid_size // 2) * (grid_size // 2)

        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(-1, 1, self.grid_size, self.grid_size).float()
        return self.fc_layers(self.conv_layers(x)).squeeze(-1)

    def predict_with_uncertainty(self, x, n_samples=10):
        self.train()  # keep dropout on
        preds = torch.stack([self.forward(x) for _ in range(n_samples)])
        return preds.mean(0), preds.std(0)
