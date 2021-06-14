import click
import random as rd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchtext.vocab as vocab
from statistics import mean
from sklearn.model_selection import train_test_split
from copy import copy
from functools import partial
from data import Task1Dataset, enrich
from analogy_reg import AnalogyRegression
from cnn_embeddings import CNNEmbedding
from store_cnn_embeddings import generate_embeddings_file
from utils import elapsed_timer, collate

@click.command()
@click.option('--language', default="arabic", prompt='The language', help='The language to evaluate the model on.', show_default=True)
@click.option('--epochs', default=20,
              help='The number of epochs the model was trained on (we use this parameter to use the right files).', show_default=True)
@click.option('--nb_analogies', default=50000, prompt='The number of analogies',
              help='The number of analogies (before augmentation) we evaluate the model on.', show_default=True)
def evaluate_solver(language, epochs, nb_analogies):
    '''Produces the accuracy for a given analogy solver.

    We look for the right vector in a given range around the closest to the produced one.

    Arguments:
    language -- The language of the model.
    epochs -- The number of epochs the models were trained on (we use this parameter to use the right files).
    nb_analogies -- The maximum number of analogies (before augmentation) we evaluate the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.'''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Test models ---

    # Store the embeddings in a file
    path_embed = f"models/classification_cnn/classification_CNN_{language}_{epochs}e.pth"
    custom_embeddings_file = generate_embeddings_file(language = language, path_embed = path_embed)

    custom_embeddings = vocab.Vectors(name = custom_embeddings_file,
                                      cache = 'embeddings/char_cnn',
                                      unk_init = torch.Tensor.normal_)

    custom_embeddings.vectors = custom_embeddings.vectors.to(device)

    # Evaluate
    train_dataset = Task1Dataset(language=language, mode="train", word_encoding="char")
    if language == "japanese":
        japanese_train_analogies, japanese_test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = 42)

        test_dataset = copy(train_dataset)
        test_dataset.analogies = japanese_test_analogies

        train_dataset.analogies = japanese_train_analogies
    else:
        test_dataset = Task1Dataset(language=language, mode="test", word_encoding="char")

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

    test_dataloader = DataLoader(test_subset, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))

    path_embed = f"models/classification_cnn/classification_CNN_{language}_{epochs}e.pth"
    path_regression = f"models/regression/regression_cnn_{language}_{epochs}e.pth"

    saved_data_embed = torch.load(path_embed)
    saved_data_regression = torch.load(path_regression)

    if language == "japanese":
        emb_size = 512
    else:
        emb_size = 64


    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)
    embedding_model.load_state_dict(saved_data_embed['state_dict_embeddings'])
    embedding_model.eval()

    regression_model = AnalogyRegression(emb_size=16*5)
    regression_model.load_state_dict(saved_data_regression['state_dict'])
    regression_model.eval()

    regression_model.to(device)
    embedding_model.to(device)

    # Cosine distance
    stored_lengths = torch.sqrt((custom_embeddings.vectors ** 2).sum(dim=1))

    def closest_cosine(vec):
        numerator = (custom_embeddings.vectors * vec).sum(dim=1)
        denominator = stored_lengths * torch.sqrt((vec ** 2).sum())
        similarities = numerator / denominator
        return custom_embeddings.itos[similarities.argmax()]

    # Euclidian distance
    def closest_euclid(vec):
        dists = torch.sqrt(((custom_embeddings.vectors - vec) ** 2).sum(dim=1))
        return custom_embeddings.itos[dists.argmin()]

    regression_model.to(device)
    embedding_model.to(device)

    accuracy_cosine = []
    accuracy_euclid = []

    with elapsed_timer() as elapsed:

        for a, b, c, d in test_dataloader:

            # compute the embeddings

            a = embedding_model(a.to(device))
            b = embedding_model(b.to(device))
            c = embedding_model(c.to(device))
            d = embedding_model(d.to(device))

            data = torch.stack([a, b, c, d], dim = 1)

            for a, b, c, d_expected in enrich(data):

                d_pred = regression_model(a, b, c)
                d_closest_cosine = closest_cosine(d_pred)
                d_closest_euclid = closest_euclid(d_pred)

                d_expected_closest_cosine = closest_cosine(d_expected)
                d_expected_closest_euclid = closest_euclid(d_expected)

                accuracy_cosine.append(1 if d_expected_closest_cosine == d_closest_cosine else 0)
                accuracy_euclid.append(1 if d_expected_closest_euclid == d_closest_euclid else 0)

    print(f'Accuracy with Cosine similarity: {mean(accuracy_cosine)}\nAccuracy with Euclidean distance: {mean(accuracy_euclid)}\n\n')

if __name__ == '__main__':
    evaluate_solver()
