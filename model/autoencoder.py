import torch.nn as nn
import math


class Autoencoder(nn.Module):

    def __init__(self, input_feature=1024, output_dimensions = 256):
        super(Autoencoder, self).__init__()        
        self.encoder = nn.Linear(input_feature, output_dimensions)
        self.decoder = nn.Linear(output_dimensions,input_feature)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

