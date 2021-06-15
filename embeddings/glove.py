import torch

GLOVE_PATH = "embeddings/glove/german.txt"
class GloVe:
    def __init__(self, path=GLOVE_PATH, dim=300):
        self.embeddings = dict()
        self.dim = dim
        with open(path, 'r') as f:
            for line in f:
                word, values = line.split(" ", 1)
                values = [float(value) for value in values.split(" ")]
                self.embeddings[word] = torch.tensor(values)
        #self.words = {v: k for k,v in self.embeddings.items()}
        #self.vectors = torch.cat(list(self.embeddings.values()), dim = 0)

    def __getitem__(self, index):
        return self.embeddings.get(index, torch.zeros(self.dim))
