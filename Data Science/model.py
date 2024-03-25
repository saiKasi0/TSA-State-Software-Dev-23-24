import torch.nn as nn
from torch.optim import Adam
import lightning as L

class LightningLSTM(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1)

    def forward(self, input):
        input_trans = input.view(len(input), 1)
        lstm_out, temp  = self.lstm(input_trans)

        prediction = lstm_out[-1]
        return prediction 
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 0.1)
    
    def training_step(self, batch):
        input_i, label_i = batch
        output_i =  self.forward(input_i[0])
        loss = (output_i - label_i)**2 # Sum of Squared Residuals
        self.log("train_loss", loss)

        return loss