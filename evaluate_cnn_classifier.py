import click
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random as rd
from sklearn.model_selection import train_test_split
from copy import copy
from functools import partial
from statistics import mean
from utils import elapsed_timer, collate, get_accuracy_classification
from data import Task1Dataset, enrich, generate_negative
from analogy_classif import Classification
from cnn_embeddings import CNNEmbedding

@click.command()
@click.option('--language', default="arabic", prompt='The language', help='The language to evaluate the model on.', show_default=True)
@click.option('--epochs', default=20,
              help='The number of epochs the model was trained on (we use this parameter to use the right files).', show_default=True)
@click.option('--nb_analogies', default=50000, prompt='The number of analogies',
              help='The maximum number of analogies (before augmentation) we evaluate the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)
def evaluate_classifier(language, epochs, nb_analogies):
    '''Produces the accuracy for valid analogies, invalid analogies for a given model.

    Arguments:
    language -- The language of the model.
    nb_analogies -- The maximum number of analogies (before augmentation) we evaluate the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.
    epochs -- The number of epochs the models were trained on (we use this parameter to use the right files).'''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = Task1Dataset(language=language, mode="train", word_encoding="char")
    if language == "japanese":
        japanese_train_analogies, japanese_test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = 42)

        test_dataset = copy(train_dataset)
        test_dataset.analogies = japanese_test_analogies

        train_dataset.analogies = japanese_train_analogies
    else:
        test_dataset = Task1Dataset(language=language, mode="test", word_encoding="char")

    # Even the char:int dictionaries
    voc = train_dataset.word_voc_id
    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value
    test_dataset.word_voc = train_dataset.word_voc
    test_dataset.word_voc_id = voc

    if len(test_dataset) > nb_analogies:
        test_indices = list(range(len(test_dataset)))
        test_sub_indices = rd.sample(test_indices, nb_analogies)
        test_subset = Subset(test_dataset, test_sub_indices)
    else:
        test_subset = test_dataset

    test_dataloader = DataLoader(test_subset, shuffle = True, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))

    path_models = f"models/classification_cnn/classification_CNN_{language}_{epochs}e.pth"
    saved_models = torch.load(path_models)

    if language == "japanese":
        emb_size = 512
    else:
        emb_size = 64

    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)
    embedding_model.load_state_dict(saved_models['state_dict_embeddings'])
    embedding_model.eval()

    classification_model = Classification(emb_size=16*5)
    classification_model.load_state_dict(saved_models['state_dict_classification'])
    classification_model.eval()

    embedding_model.to(device)
    classification_model.to(device)

    accuracy_true = []
    accuracy_false = []

    for a, b, c, d in test_dataloader:

        # compute the embeddings
        a = embedding_model(a)
        b = embedding_model(b)
        c = embedding_model(c)
        d = embedding_model(d)

        data = torch.stack([a, b, c, d], dim = 1)

        for a, b, c, d in enrich(data):

            # positive example, target is 1
            a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
            b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
            c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
            d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

            is_analogy = torch.squeeze(classification_model(a, b, c, d))

            expected = torch.ones(is_analogy.size(), device=is_analogy.device)

            accuracy_true.append(get_accuracy_classification(expected, is_analogy))


        for a, b, c, d in generate_negative(data):

            # negative examples, target is 0
            a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
            b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
            c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
            d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

            is_analogy = torch.squeeze(classification_model(a, b, c, d))

            expected = torch.zeros(is_analogy.size(), device=is_analogy.device)

            accuracy_false.append(get_accuracy_classification(expected, is_analogy))

    print(f'Accuracy for valid analogies: {mean(accuracy_true)}\nAccuracy for invalid analogies: {mean(accuracy_false)}')

if __name__ == '__main__':
    evaluate_classifier()
