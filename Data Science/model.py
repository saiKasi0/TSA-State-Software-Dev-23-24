import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

class LSTMbyHand(L.LightningModule):
    
    # Create & Init Weight and Bias Tensors
    def __init__(self):

        super().__init__()
        mean  = torch.tensor(0.0)
        std =  torch.tensor(1.0)

        #  Initially random weights and biases for LSTM 
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    # Handle LSTM Operations
    def lstm_unit(self, input_value, long_memory, short_memory):
        # Forget Gate: Calculates How Much of Long Term Memory to Implement
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + (input_value * self.wlr2) + self.blr1)

        # Input Gate: Creates New Potential Long-term Memory and Determines How Much To Remember
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + (input_value * self.wpr2) + self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) + (input_value * self.wp2) + self.bp1)

        # Updates Long Term Memory
        updated_long_memory = ((long_memory * long_remember_percent) + (potential_remember_percent * potential_memory))
        
        # Output Gate: Makes New Short Term Memory and Determines How Much To Remember
        output_percent = torch.sigmoid((short_memory * self.wo1) + (input_value * self.wo2) + self.bo1)
        updated_short_memory =  torch.tanh(updated_long_memory) * output_percent

        return ([updated_long_memory, updated_short_memory])

    # Conducts Forward Pass in Unrolled LSTM
    def forward(self, input):
        long_memory = 0
        short_memory = 0
        for data in input:
            long_memory, short_memory = self.lstm_unit(data, long_memory, short_memory)
        return short_memory

    # Configures Adam Optimizer
    def configure_optimizers(self):
        return Adam(self.parameters)

    # Calculate Loss (Sum of Squared Residuals) and Log Training Progress
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i =  self.forward(input_i[0])
        loss = (output_i - label_i)**2 # Sum of Squared Residuals
        self.log("train_loss", loss)

        return loss