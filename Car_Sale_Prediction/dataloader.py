import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self, path, split='train', test_size=0.2, valid_size=0.2, random_state=42, encoding='ISO-8859-1'):
        # Load and preprocess data
        self.df = pd.read_csv(path, encoding=encoding)
        self.df = self.df.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1)

        # Normalize data
        scaler = MinMaxScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)

        # Split into features (X) and target (y)
        X = self.df.drop('Car Purchase Amount', axis=1)
        y = self.df['Car Purchase Amount']

        # Split train/test first
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Further split train into train/valid
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=random_state)

        # Assign data based on the split
        if split == 'train':
            self.X = torch.tensor(X_train.values, dtype=torch.float32)
            self.y = torch.tensor(y_train.values, dtype=torch.float32)
        elif split == 'valid':
            self.X = torch.tensor(X_valid.values, dtype=torch.float32)
            self.y = torch.tensor(y_valid.values, dtype=torch.float32)
        elif split == 'test':
            self.X = torch.tensor(X_test.values, dtype=torch.float32)
            self.y = torch.tensor(y_test.values, dtype=torch.float32)
        else:
            raise ValueError("Invalid split. Choose from 'train', 'valid', or 'test'.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]