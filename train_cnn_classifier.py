import click
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random as rd
from functools import partial
from statistics import mean
from sklearn.model_selection import train_test_split
from copy import copy
from utils import elapsed_timer, collate
from data import Task1Dataset, enrich, generate_negative
from analogy_classif import Classification
from cnn_embeddings import CNNEmbedding


@click.command()
@click.option('--language', default="arabic", prompt='The language', help='The language to train the model on.', show_default=True)
@click.option('--nb_analogies', default=50000, prompt='The number of analogies',
              help='The maximum number of analogies (before augmentation) we train the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)
@click.option('--epochs', default=20, prompt='The number of epochs',
              help='The number of epochs we train the model for.', show_default=True)
def train_classifier(language, nb_analogies, epochs):
    '''Trains a classifier and a word embedding model for a given language.

    Arguments:
    language -- The language of the data to use for the training.
    nb_analogies -- The number of analogies to use (before augmentation) for the training.
    epochs -- The number of epochs we train the model for.'''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Prepare data ---

    ## Train and test dataset
    train_dataset = Task1Dataset(language=language, mode="train", word_encoding="char")

    if language == "japanese":
        japanese_train_analogies, japanese_test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = 42)

        #japanese_test_dataset = copy(train_dataset)
        #japanese_test_dataset.analogies = japanese_test_analogies

        train_dataset.analogies = japanese_train_analogies

    voc = train_dataset.word_voc_id
    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

    # Get subsets

    if len(train_dataset) > nb_analogies:
        train_indices = list(range(len(train_dataset)))
        train_sub_indices = rd.sample(train_indices, nb_analogies)
        train_subset = Subset(train_dataset, train_sub_indices)
    else:
        train_subset = train_dataset


    # Load data
    train_dataloader = DataLoader(train_subset, shuffle = True, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))


    # --- Training models ---
    if language == 'japanese':
        emb_size = 512
    else:
        emb_size = 64

    classification_model = Classification(emb_size=16*5) # 16 because 16 filters of each size, 5 because 5 sizes
    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)

    # --- Training Loop ---
    classification_model.to(device)
    embedding_model.to(device)

    optimizer = torch.optim.Adam(list(classification_model.parameters()) + list(embedding_model.parameters()))
    criterion = nn.BCELoss()

    losses_list = []
    times_list = []

    for epoch in range(epochs):

        losses = []
        with elapsed_timer() as elapsed:
            for a, b, c, d in train_dataloader:

                optimizer.zero_grad()

                # compute the embeddings
                a = embedding_model(a.to(device))
                b = embedding_model(b.to(device))
                c = embedding_model(c.to(device))
                d = embedding_model(d.to(device))

                # to be able to add other losses, which are tensors, we initialize the loss as a 0 tensor
                loss = torch.tensor(0).to(device).float()

                data = torch.stack([a, b, c, d], dim = 1)

                for a, b, c, d in enrich(data):

                    # positive example, target is 1
                    a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                    b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                    c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                    d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                    is_analogy = classification_model(a, b, c, d)

                    expected = torch.ones(is_analogy.size(), device=is_analogy.device)

                    loss += criterion(is_analogy, expected)

                for a, b, c, d in generate_negative(data):

                    # negative examples, target is 0
                    a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                    b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                    c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                    d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                    is_analogy = classification_model(a, b, c, d)

                    expected = torch.zeros(is_analogy.size(), device=is_analogy.device)

                    loss += criterion(is_analogy, expected)

                # once we have all the losses for one set of embeddings, we can backpropagate
                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().item())

        losses_list.append(mean(losses))
        times_list.append(elapsed())
        print(f"Epoch: {epoch}, Run time: {times_list[-1]:4.5}s, Loss: {losses_list[-1]}")

    torch.save({"state_dict_classification": classification_model.cpu().state_dict(), "state_dict_embeddings": embedding_model.cpu().state_dict(), "losses": losses_list, "times": times_list}, f"models/classification_cnn/classification_CNN_{language}_{epochs}e.pth")

if __name__ == '__main__':
    train_classifier()
