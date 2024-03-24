import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam

from torch.utils.data import TensorDataset, DataLoader

class LSTMbyHand():
    # Create & Init Weight and Bias Tensors
    def __init__(self):
        pass

    # Handle LSTM Operations
    def lstm_unit(self, input_value, long_memort, short_memory):
        pass

    # Forward pass unrolled LSTM
    def forward(self, input):
        pass

    # Configures Adam Optimizer
    def configure_optimizers(self):
        pass

    # Calculate Loss (Sum of Squared Residuals) and Log Training Progress
    def training_step(self, batch, batch_idx):
        pass
