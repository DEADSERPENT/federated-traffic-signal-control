"""
Traffic Signal Optimization Model
Neural network model for predicting optimal green signal duration.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from collections import OrderedDict


class TrafficSignalModel(nn.Module):
    """
    Neural network model for traffic signal optimization.

    Input features:
    - Queue length for 4 directions (north, south, east, west)
    - Current phase (encoded as 0/1)
    - Current green duration (normalized)

    Output:
    - Predicted optimal green duration
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_layers: List[int] = None,
        output_dim: int = 1
    ):
        """
        Initialize the model.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            output_dim: Number of output values
        """
        super(TrafficSignalModel, self).__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32]

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append((f"linear_{i}", nn.Linear(prev_dim, hidden_dim)))
            layers.append((f"relu_{i}", nn.ReLU()))
            layers.append((f"dropout_{i}", nn.Dropout(0.2)))
            prev_dim = hidden_dim

        # Output layer
        layers.append(("output", nn.Linear(prev_dim, output_dim)))

        self.network = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on numpy array input.

        Args:
            features: Input features as numpy array

        Returns:
            Predicted optimal green duration
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            predictions = self.forward(x)
            return predictions.numpy().flatten()

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as list of numpy arrays."""
        return [param.cpu().detach().numpy() for param in self.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from list of numpy arrays."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in params_dict
        })
        self.load_state_dict(state_dict, strict=True)


class LinearRegressionModel(nn.Module):
    """Simple linear regression model for baseline comparison."""

    def __init__(self, input_dim: int = 6, output_dim: int = 1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_parameters(self) -> List[np.ndarray]:
        return [param.cpu().detach().numpy() for param in self.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in params_dict
        })
        self.load_state_dict(state_dict, strict=True)


def create_model(model_type: str = "neural_network", **kwargs) -> nn.Module:
    """
    Factory function to create traffic signal models.

    Args:
        model_type: "neural_network" or "linear_regression"
        **kwargs: Additional arguments for model initialization

    Returns:
        PyTorch model instance
    """
    if model_type == "neural_network":
        return TrafficSignalModel(**kwargs)
    elif model_type == "linear_regression":
        return LinearRegressionModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    model: nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.01
) -> Tuple[nn.Module, List[float]]:
    """
    Train the model on local data.

    Args:
        model: PyTorch model to train
        train_data: Tuple of (features, labels)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate

    Returns:
        Tuple of (trained model, loss history)
    """
    features, labels = train_data
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

    return model, loss_history


def evaluate_model(
    model: nn.Module,
    test_data: Tuple[np.ndarray, np.ndarray]
) -> Tuple[float, float]:
    """
    Evaluate the model on test data.

    Args:
        model: PyTorch model to evaluate
        test_data: Tuple of (features, labels)

    Returns:
        Tuple of (MSE loss, MAE)
    """
    features, labels = test_data
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        predictions = model(features)
        mse = nn.MSELoss()(predictions, labels).item()
        mae = nn.L1Loss()(predictions, labels).item()

    return mse, mae


if __name__ == "__main__":
    # Test the model
    print("Testing Traffic Signal Model...")

    # Create model
    model = create_model("neural_network", hidden_layers=[64, 32])
    print(f"Model architecture:\n{model}")

    # Generate dummy data
    np.random.seed(42)
    features = np.random.rand(100, 6).astype(np.float32)
    labels = np.random.rand(100).astype(np.float32) * 60 + 20  # 20-80 seconds

    # Train
    print("\nTraining model...")
    model, losses = train_model(model, (features, labels), epochs=5)
    print(f"Final loss: {losses[-1]:.4f}")

    # Evaluate
    mse, mae = evaluate_model(model, (features[:20], labels[:20]))
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Test prediction
    sample_features = features[0]
    prediction = model.predict(sample_features)
    print(f"\nSample prediction: {prediction[0]:.2f} seconds")
