"""
Traffic Signal Optimization Model
Neural network model for predicting optimal green signal duration.
Optimized architecture for Federated Learning performance.

GPU-Agnostic: Automatically uses GPU when available, falls back to CPU.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from collections import OrderedDict
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.device import get_device, to_device


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
        dropout_rate: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the model.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            output_dim: Number of output values
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            device: Device to use (auto-detected if None)
        """
        super(TrafficSignalModel, self).__init__()

        # Set device - auto-detect if not specified
        self._device = device if device is not None else get_device()

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

        # Move model to device
        self.to(self._device)

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
            x = torch.FloatTensor(features).to(self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            predictions = self.forward(x)
            return predictions.cpu().numpy().flatten()

    def get_parameters(self) -> List[np.ndarray]:
        """Get model state dict values as list of numpy arrays (includes BatchNorm buffers)."""
        return [val.cpu().detach().numpy() for val in self.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model state from list of numpy arrays (includes BatchNorm buffers)."""
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key, param in zip(keys, parameters):
            state_dict[key] = torch.tensor(param, device=self._device)
        self.load_state_dict(state_dict, strict=True)

    def get_state_keys(self) -> List[str]:
        """Get state dict keys for debugging."""
        return list(self.state_dict().keys())


class LinearRegressionModel(nn.Module):
    """Simple linear regression model for baseline comparison."""

    def __init__(self, input_dim: int = 6, output_dim: int = 1, device: Optional[torch.device] = None):
        super(LinearRegressionModel, self).__init__()
        self._device = device if device is not None else get_device()
        self.linear = nn.Linear(input_dim, output_dim)
        self.to(self._device)

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
            state_dict[key] = torch.tensor(param, device=self._device)
        self.load_state_dict(state_dict, strict=True)


class LSTMTrafficModel(nn.Module):
    """
    LSTM-based model for time-series traffic prediction.

    Captures temporal dependencies in traffic patterns:
    - Queue length at time t depends on queue length at t-1, t-2, etc.
    - Rush hour patterns, periodic fluctuations
    - Event-driven traffic spikes

    Input: Sequence of traffic states [batch, seq_len, features]
    Output: Predicted optimal green duration
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_rate: float = 0.1,
        bidirectional: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize LSTM model.

        Args:
            input_dim: Number of input features per timestep
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of stacked LSTM layers
            output_dim: Output dimension (1 for green duration)
            dropout_rate: Dropout for regularization
            bidirectional: Use bidirectional LSTM
            device: Device to use
        """
        super(LSTMTrafficModel, self).__init__()

        self._device = device if device is not None else get_device()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Fully connected layers after LSTM
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

        self._initialize_weights()
        self.to(self._device)

    def _initialize_weights(self):
        """Initialize LSTM and FC weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, features] or [batch, features]

        Returns:
            Predicted green duration [batch, 1]
        """
        # Handle non-sequence input (single timestep)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state (or concatenated for bidirectional)
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]

        # FC layers
        output = self.fc(hidden)
        return output

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on numpy array input."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            predictions = self.forward(x)
            return predictions.cpu().numpy().flatten()

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as list of numpy arrays."""
        return [val.cpu().detach().numpy() for val in self.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from list of numpy arrays."""
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key, param in zip(keys, parameters):
            state_dict[key] = torch.tensor(param, device=self._device)
        self.load_state_dict(state_dict, strict=True)


class GRUTrafficModel(nn.Module):
    """
    GRU-based model for time-series traffic prediction.

    Similar to LSTM but with fewer parameters:
    - Faster training, lower memory usage
    - Often performs comparably to LSTM on shorter sequences
    - Better for edge deployment (smaller model size)

    Input: Sequence of traffic states [batch, seq_len, features]
    Output: Predicted optimal green duration
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_rate: float = 0.1,
        bidirectional: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize GRU model.

        Args:
            input_dim: Number of input features per timestep
            hidden_dim: GRU hidden state dimension
            num_layers: Number of stacked GRU layers
            output_dim: Output dimension (1 for green duration)
            dropout_rate: Dropout for regularization
            bidirectional: Use bidirectional GRU
            device: Device to use
        """
        super(GRUTrafficModel, self).__init__()

        self._device = device if device is not None else get_device()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Fully connected layers after GRU
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

        self._initialize_weights()
        self.to(self._device)

    def _initialize_weights(self):
        """Initialize GRU and FC weights."""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, features] or [batch, features]

        Returns:
            Predicted green duration [batch, 1]
        """
        # Handle non-sequence input (single timestep)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # GRU forward pass
        gru_out, h_n = self.gru(x)

        # Use last hidden state (or concatenated for bidirectional)
        if self.bidirectional:
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]

        # FC layers
        output = self.fc(hidden)
        return output

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on numpy array input."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            predictions = self.forward(x)
            return predictions.cpu().numpy().flatten()

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as list of numpy arrays."""
        return [val.cpu().detach().numpy() for val in self.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from list of numpy arrays."""
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key, param in zip(keys, parameters):
            state_dict[key] = torch.tensor(param, device=self._device)
        self.load_state_dict(state_dict, strict=True)


def create_model(
    model_type: str = "neural_network",
    optimized: bool = True,
    device: Optional[torch.device] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create traffic signal models.

    Args:
        model_type: Model architecture type
            - "neural_network" or "mlp": Standard MLP (default)
            - "lstm": LSTM for time-series (captures temporal dependencies)
            - "gru": GRU for time-series (lighter than LSTM)
            - "linear_regression": Simple baseline
        optimized: Use optimized architecture for FL (deeper network)
        device: Device to use (auto-detected if None)
        **kwargs: Additional arguments for model initialization

    Returns:
        PyTorch model instance on the specified device
    """
    # Get device if not specified
    if device is None:
        device = get_device()

    if model_type in ["neural_network", "mlp"]:
        # Use optimized architecture by default
        if optimized and "hidden_layers" not in kwargs:
            kwargs["hidden_layers"] = [128, 64, 32]
        if "use_batch_norm" not in kwargs:
            kwargs["use_batch_norm"] = True
        if "dropout_rate" not in kwargs:
            kwargs["dropout_rate"] = 0.1
        return TrafficSignalModel(device=device, **kwargs)

    elif model_type == "lstm":
        # LSTM for time-series traffic prediction
        if "hidden_dim" not in kwargs:
            kwargs["hidden_dim"] = 64
        if "num_layers" not in kwargs:
            kwargs["num_layers"] = 2
        if "dropout_rate" not in kwargs:
            kwargs["dropout_rate"] = 0.1
        return LSTMTrafficModel(device=device, **kwargs)

    elif model_type == "gru":
        # GRU for time-series (lighter alternative to LSTM)
        if "hidden_dim" not in kwargs:
            kwargs["hidden_dim"] = 64
        if "num_layers" not in kwargs:
            kwargs["num_layers"] = 2
        if "dropout_rate" not in kwargs:
            kwargs["dropout_rate"] = 0.1
        return GRUTrafficModel(device=device, **kwargs)

    elif model_type == "linear_regression":
        return LinearRegressionModel(device=device, **kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: neural_network, mlp, lstm, gru, linear_regression")


def train_model(
    model: nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    use_scheduler: bool = True,
    gradient_clip: float = 1.0,
    global_model: nn.Module = None,
    mu: float = 0.01,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, List[float]]:
    """
    Enhanced training with FedProx support for better FL performance.

    FedProx adds a proximal term that prevents local models from drifting
    too far from the global model, improving convergence on Non-IID data.

    Args:
        model: PyTorch model to train
        train_data: Tuple of (features, labels)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: L2 regularization weight decay
        use_scheduler: Whether to use learning rate scheduler
        gradient_clip: Max gradient norm for clipping
        global_model: Global model for FedProx (None for standard training)
        mu: FedProx proximal term weight (0.01-0.1 recommended)
        device: Device to use (auto-detected if None)

    Returns:
        Tuple of (trained model, loss history)
    """
    # Get device from model or auto-detect
    if device is None:
        if hasattr(model, '_device'):
            device = model._device
        else:
            device = get_device()

    features, labels = train_data
    features = torch.FloatTensor(features).to(device)
    labels = torch.FloatTensor(labels).unsqueeze(1).to(device)

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

    # Cosine annealing scheduler - keep learning active longer
    scheduler = None
    if use_scheduler and epochs > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.1
        )

    # Freeze global model parameters for FedProx
    global_params = None
    if global_model is not None and mu > 0:
        global_params = [p.clone().detach().to(device) for p in global_model.parameters()]

    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)

            # FedProx: Add proximal term to prevent drift from global model
            if global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(model.parameters(), global_params):
                    proximal_term += torch.sum((local_param - global_param) ** 2)
                loss = loss + (mu / 2.0) * proximal_term

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
    test_data: Tuple[np.ndarray, np.ndarray],
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """
    Evaluate the model on test data.

    Args:
        model: PyTorch model to evaluate
        test_data: Tuple of (features, labels)
        device: Device to use (auto-detected if None)

    Returns:
        Tuple of (MSE loss, MAE)
    """
    # Get device from model or auto-detect
    if device is None:
        if hasattr(model, '_device'):
            device = model._device
        else:
            device = get_device()

    features, labels = test_data
    features = torch.FloatTensor(features).to(device)
    labels = torch.FloatTensor(labels).unsqueeze(1).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(features)
        mse = nn.MSELoss()(predictions, labels).item()
        mae = nn.L1Loss()(predictions, labels).item()

    return mse, mae


if __name__ == "__main__":
    # Test the enhanced model with GPU/CPU auto-detection
    print("="*60)
    print("Testing Enhanced Traffic Signal Model with GPU/CPU Support")
    print("="*60)

    # Show device info
    device = get_device()
    print(f"\nUsing device: {device}")

    # Create optimized model (automatically on best device)
    model = create_model("neural_network", hidden_layers=[128, 64, 32])
    print(f"\nModel architecture:\n{model}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model device: {model._device}")

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

    print("\n" + "="*60)
    print("GPU/CPU support test completed successfully!")
    print("="*60)
