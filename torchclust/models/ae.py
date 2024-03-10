import torch
import torch.nn as nn

class AutoEncoder(torch.nn.Module):

    def __init__(self, hidden_units):
        super(AutoEncoder, self).__init__()
        self.hidden_units = hidden_units
        n_stacks = len(self.hidden_units) - 1

        ### ENCODER
        encoder_layers = []
        decoder_layers = []
        for i in range(n_stacks - 1):
            layer = nn.Linear(self.hidden_units[i], self.hidden_units[i + 1])
            encoder_layers.append(layer)
            encoder_layers.append(nn.LeakyReLU())
        encoder_layers.append(nn.Linear(self.hidden_units[-2], self.hidden_units[-1]))

        ### DECODER
        for i in range(n_stacks, 1, -1):
            layer = nn.Linear(self.hidden_units[i], self.hidden_units[i - 1])
            decoder_layers.append(layer)
            decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.Linear(self.hidden_units[1], self.hidden_units[0]))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward_encoder(self, x):
        z = self.encoder(x)
        return z
    
    def forward_decoder(self, z):
        xx = self.decoder(z)
        return xx
        

    def forward(self, x):
        
        ### ENCODER
        encoded = self.forward_encoder(x)
        
        ### DECODER
        decoded = self.forward_decoder(encoded)
        
        return decoded