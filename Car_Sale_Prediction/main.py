import torch
from model import MyModel
from dataloader import MyDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tqdm import tqdm
import numpy as np

def train(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        # Training Loop with TQDM
        loop = tqdm(enumerate(train_loader), total=len(train_loader), unit="batch", desc=f"Epoch [{epoch}/{epochs}]")
        for batch, (X, y) in loop:
            X, y = X.to(device), y.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X).squeeze(-1)
            loss = criterion(outputs, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update progress bar for training
            avg_train_loss = train_loss / (batch + 1)
            loop.set_postfix({"Train Loss": avg_train_loss})

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model with metrics: MAE, MSE, RMSE.
    
    Args:
        model: The model to evaluate.
        data_loader: DataLoader for the test set.
        device: Device ('cuda' or 'cpu').

    Returns:
        metrics: Dictionary with MAE, MSE, and RMSE values.
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # Predict
            outputs = model(X).squeeze(-1)  # Remove unnecessary dimension if any
            predictions.extend(outputs.cpu().numpy())
            targets.extend(y.cpu().numpy())

    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

    # Print results
    print("Evaluation results:")
    print(f"MAE: {mae:.2f} - MSE: {mse:.2f} - RMSE: {rmse:.2f}")

    return metrics


def main():
    # File path to data
    path = 'Car_Purchasing_Data.csv'

    # Create datasets and loaders
    train_dataset = MyDataset(path, split='train')
    valid_dataset = MyDataset(path, split='valid')
    test_dataset = MyDataset(path, split='test')

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model, loss, and optimizer
    model = MyModel().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Train the model
    epochs = 10
    train(model, train_loader, valid_loader, criterion, optimizer, epochs, device)

    # Evaluate the model on the test set
    evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    main() 