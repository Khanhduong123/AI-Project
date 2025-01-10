import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 25)  # input_dim=5, output_dim=25
        self.fc2 = nn.Linear(25, 25) # hidden layer: input_dim=25, output_dim=25
        self.fc3 = nn.Linear(25, 1)  # output_dim=1
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # The last layer applies linear activation
        return x

