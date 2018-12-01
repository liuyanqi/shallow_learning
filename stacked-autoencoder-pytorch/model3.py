import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(100, 150, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(150, 200, kernel_size=3, stride=1, padding=1)
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(200, 100, kernel_size=2, stride=2, padding=0), 
            nn.ConvTranspose2d(100, 3, kernel_size=2, stride=2, padding=0), 
            )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.5)
        self.epoch = -1

    def forward(self, x, epoch):
        y = self.encoder(x)
        if self.training:
            x_reconstruct = self.decoder(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.epoch != epoch:
                print(loss.data[0])
                self.epoch = epoch
        return y

    def reconstruct(self,x):
        return self.decoder(x)