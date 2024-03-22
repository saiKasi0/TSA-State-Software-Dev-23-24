import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# So far only have basic NN stuff need to implemt LSTM and add layers

# Recursive Nerual Net Model
class RNN_Model(nn.Module):

    # Instation
    def __init__(self):
        super().__init__()
        # TODO
        pass

    #  Basic forward operation #TODO ask Darth if this is alr
    def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    # Save data of training into a pth file (helpful to show training progression) : for memmory
    def save(self, file_name='model.pth'): #TODO might want to add like a name for each model here
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# Model trainer
class RNNTranier:
    # Basic innit subject to change
    def __init__(self, model, lr, gamma):
        self.lr = lr # learning rate
        self.gamma = gamma # influence of single training exmaple
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Mean Squared Regression Criterion

    # TODO
    def train_step(self):
        pass
