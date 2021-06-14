import torch, torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from statistics import mean

class CNNEmbedding(nn.Module):
    def __init__(self, emb_size, voc_size):
        ''' Character level CNN word embedding.

        It produces an output of length 80 by applying filters of different sizes on the input.
        It uses 16 filters for each size from 2 to 6.

        Arguments:
        emb_size -- the size of the input vectors
        voc_size -- the maximum number to find in the input vectors
        '''
        super().__init__()

        self.emb_size = emb_size
        self.voc_size = voc_size

        self.embedding = nn.Embedding(voc_size, emb_size)

        self.conv2 = nn.Conv1d(emb_size, 16, 2, padding = 0)
        self.conv3 = nn.Conv1d(emb_size, 16, 3, padding = 0)
        self.conv4 = nn.Conv1d(emb_size, 16, 4, padding = 1)
        self.conv5 = nn.Conv1d(emb_size, 16, 5, padding = 2)
        self.conv6 = nn.Conv1d(emb_size, 16, 6, padding = 3)


    def forward(self, word):
        # Embedds the word and set the right dimension for the tensor
        unk = word<0
        word[unk] = 0
        word = self.embedding(word)
        word[unk] = 0
        word = torch.transpose(word, 1,2)

        # Apply each conv layer -> torch.Size([batch_size, 16, whatever])
        size2 = self.conv2(word)
        size3 = self.conv3(word)
        size4 = self.conv4(word)
        size5 = self.conv5(word)
        size6 = self.conv6(word)

        # Get the max of each channel -> torch.Size([batch_size, 16])
        maxima2 = torch.max(size2, dim = -1)
        maxima3 = torch.max(size3, dim = -1)
        maxima4 = torch.max(size4, dim = -1)
        maxima5 = torch.max(size5, dim = -1)
        maxima6 = torch.max(size6, dim = -1)

        # Concatenate the 5 vectors to get 1 -> torch.Size([batch_size, 80])
        output = torch.cat([maxima2[0], maxima3[0], maxima4[0], maxima5[0], maxima6[0]], dim = -1)

        return output



