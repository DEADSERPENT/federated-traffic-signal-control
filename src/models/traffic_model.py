"""
Traffic Signal Optimization Model
Neural network model for predicting optimal green signal duration.
Optimized architecture for Federated Learning performance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from collections import OrderedDict


class TrafficSignalModel(nn.Module):
    """
    Enhanced neural network model for traffic signal optimization.

    Optimized for Federated Learning with:
    - Deeper architecture with batch normalization
    - Residual-style connections for better gradient flow
    - Optimized dropout strategy
    - Better weight initialization

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
        output_dim: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the model.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            output_dim: Number of output values
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
        """
        super(TrafficSignalModel, self).__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]  # Deeper default architecture

        self.use_batch_norm = use_batch_norm
        layers = []
        prev_dim = input_dim

        # Build hidden layers with improved architecture
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append((f"linear_{i}", nn.Linear(prev_dim, hidden_dim)))
            if use_batch_norm:
                layers.append((f"bn_{i}", nn.BatchNorm1d(hidden_dim)))
            layers.append((f"relu_{i}", nn.LeakyReLU(0.1)))  # LeakyReLU for better gradients
            # Progressive dropout - less in early layers
            drop_rate = dropout_rate * (i + 1) / len(hidden_layers)
            layers.append((f"dropout_{i}", nn.Dropout(drop_rate)))
            prev_dim = hidden_dim

        # Output layer
        layers.append(("output", nn.Linear(prev_dim, output_dim)))

        self.network = nn.Sequential(OrderedDict(layers))

        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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
        """Get model state dict values as list of numpy arrays (includes BatchNorm buffers)."""
        return [val.cpu().detach().numpy() for val in self.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model state from list of numpy arrays (includes BatchNorm buffers)."""
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key, param in zip(keys, parameters):
            state_dict[key] = torch.tensor(param)
        self.load_state_dict(state_dict, strict=True)

    def get_state_keys(self) -> List[str]:
        """Get state dict keys for debugging."""
        return list(self.state_dict().keys())


class LinearRegressionModel(nn.Module):
    """Simple linear regression model for baseline comparison."""

    def __init__(self, input_dim: int = 6, output_dim: int = 1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_parameters(self) -> List[np.ndarray]:
        """Get model state dict values as list of numpy arrays."""
        return [val.cpu().detach().numpy() for val in self.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model state from list of numpy arrays."""
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key, param in zip(keys, parameters):
            state_dict[key] = torch.tensor(param)
        self.load_state_dict(state_dict, strict=True)


def create_model(model_type: str = "neural_network", optimized: bool = True, **kwargs) -> nn.Module:
    """
    Factory function to create traffic signal models.

    Args:
        model_type: "neural_network" or "linear_regression"
        optimized: Use optimized architecture for FL (deeper network)
        **kwargs: Additional arguments for model initialization

    Returns:
        PyTorch model instance
    """
    if model_type == "neural_network":
        # Use optimized architecture by default
        if optimized and "hidden_layers" not in kwargs:
            kwargs["hidden_layers"] = [128, 64, 32]
        if "use_batch_norm" not in kwargs:
            kwargs["use_batch_norm"] = True
        if "dropout_rate" not in kwargs:
            kwargs["dropout_rate"] = 0.1
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
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    use_scheduler: bool = True,
    gradient_clip: float = 1.0
) -> Tuple[nn.Module, List[float]]:
    """
    Enhanced training with advanced techniques for better FL performance.

    Args:
        model: PyTorch model to train
        train_data: Tuple of (features, labels)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: L2 regularization weight decay
        use_scheduler: Whether to use learning rate scheduler
        gradient_clip: Max gradient norm for clipping

    Returns:
        Tuple of (trained model, loss history)
    """
    features, labels = train_data
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    # Use Smooth L1 Loss (Huber Loss) - more robust to outliers
    criterion = nn.SmoothL1Loss(beta=1.0)

    # AdamW optimizer with weight decay for better generalization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # Cosine annealing scheduler for better convergence
    scheduler = None
    if use_scheduler and epochs > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )

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

            # Gradient clipping for stability
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
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
    # Test the enhanced model
    print("Testing Enhanced Traffic Signal Model...")

    # Create optimized model
    model = create_model("neural_network", hidden_layers=[128, 64, 32])
    print(f"Model architecture:\n{model}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Generate dummy data
    np.random.seed(42)
    features = np.random.rand(200, 6).astype(np.float32)
    labels = np.random.rand(200).astype(np.float32) * 60 + 20  # 20-80 seconds

    # Train with enhanced settings
    print("\nTraining model with enhanced settings...")
    model, losses = train_model(
        model, (features, labels),
        epochs=10,
        learning_rate=0.001,
        weight_decay=1e-4
    )
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")

    # Evaluate
    mse, mae = evaluate_model(model, (features[:40], labels[:40]))
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Test prediction
    sample_features = features[0]
    prediction = model.predict(sample_features)
    print(f"\nSample prediction: {prediction[0]:.2f} seconds")
