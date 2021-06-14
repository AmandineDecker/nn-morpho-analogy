import torch, torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from statistics import mean

class Classification(nn.Module):
    def __init__(self, emb_size):
        '''CNN based analogy classifier model.

        It generates a value between 0 and 1 (0 for invalid, 1 for valid) based on four input vectors.
        1st layer (convolutional): 128 filters (= kernels) of size h × w = 1 × 2 with strides (1, 2) and relu activation.
        2nd layer (convolutional): 64 filters of size (2, 2) with strides (2, 2) and relu activation.
        3rd layer (dense, equivalent to linear for PyTorch): one output and sigmoid activation.

        Argument:
        emb_size -- the size of the input vectors'''
        super().__init__()
        self.emb_size = emb_size
        self.conv1 = nn.Conv2d(1, 128, (1,2), stride=(1,2))
        self.conv2 = nn.Conv2d(128, 64, (2,2), stride=(2,2))
        self.linear = nn.Linear(64*(emb_size//2), 1)

    def flatten(self, t):
        '''Flattens the input tensor.'''
        t = t.reshape(t.size()[0], -1)
        return t

    def forward(self, a, b, c, d):
        image = torch.stack([a, b, c, d], dim = 3)
        x = self.conv1(image)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        output = torch.sigmoid(x)
        return output
