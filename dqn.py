import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):

    def __init__(self, in_space, out_space):
        super(DQN, self).__init__()

        #create a conv block
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 18),
            nn.ReLU(),
            nn.Conv1d(32, 48, 6),
            nn.ReLU(),
            nn.Conv1d(48, 64, 3),
            nn.ReLU()
        )

        #get the conv out size
        size = self.get_conv_out_size((1,in_space))

        #create a dense block
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, out_space)
        )

    def get_conv_out_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        out = self.conv(x).view(x.shape[0],-1)
        return self.fc(out)