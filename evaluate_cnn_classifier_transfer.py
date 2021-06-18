import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import random as rd
from sklearn.model_selection import train_test_split
from functools import partial
from statistics import mean
from copy import copy
from analogy_classif import Classification
from cnn_embeddings import CNNEmbedding
from data import Task1Dataset, enrich, generate_negative
from utils import elapsed_timer, collate, get_accuracy_classification

@click.command()
@click.option('--nb_analogies', default=50000, prompt='The number of analogies',
              help='The maximum number of analogies (before augmentation) we evaluate the models on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.', show_default=True)
@click.option('--epochs', default=20,
              help='The number of epochs the models were trained on (we use this parameter to use the right files).', show_default=True)
@click.option('--mode', type=click.Choice(['full', 'partial'], case_sensitive=False), default='full', prompt='The transfer mode', help='The language of the classifier.', show_default=True)
def evaluate_classifier_transfer(nb_analogies, epochs, mode):
    '''Produces the accuracy for valid analogies, invalid analogies and analogies before data augmentation with all the models tranfered to all the languages.

    Full transfer means we use the embedding model and the classifier of one language on all the others.
    Partial transfer means we use the classifier of one language on all the others but with the right embedding model.

    Arguments:
    nb_analogies -- The maximum number of analogies (before augmentation) we evaluate the models on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.
    epochs -- The number of epochs the models were trained on (we use this parameter to use the right files).
    mode -- The transfer mode.'''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    LANGUAGES = ["arabic", "finnish", "georgian", "german", "hungarian", "japanese", "maltese", "navajo", "russian", "spanish", "turkish"]

    # Load Japanese dataset and split it into train and test
    japanese_train_dataset = Task1Dataset(language="japanese", mode="train", word_encoding="char")
    japanese_train_analogies, japanese_test_analogies = train_test_split(japanese_train_dataset.analogies, test_size=0.3, random_state = 42)

    japanese_test_dataset = copy(japanese_train_dataset)
    japanese_test_dataset.analogies = japanese_test_analogies

    japanese_train_dataset.analogies = japanese_train_analogies

    # Load the models
    dict_models = {}
    with elapsed_timer() as elapsed:
        for language in LANGUAGES:
            if language == 'japanese':
                dict_models[language] = japanese_train_dataset
            else:
                dict_models[language] = Task1Dataset(language=language, mode="train", word_encoding="char")
            #print(f"{language}: {elapsed():4.5}s")
    # Load the data
    dict_data = {}
    with elapsed_timer() as elapsed:
        for language in LANGUAGES:
            if language == 'japanese':
                dict_data[language] = japanese_test_dataset
            else:
                dict_data[language] = Task1Dataset(language=language, mode="test", word_encoding="char")
            #print(f"{language}: {elapsed():4.5}s")

    dic_of_accuracies = {}

    for l1 in LANGUAGES:

        with elapsed_timer() as elapsed:

            # Get the right place to look at for the models
            train_dataset = copy(dict_models[l1])
            path_classif = f"models/classification_cnn/classification_CNN_{l1}_{epochs}e.pth"

            list_of_accuracies = []

            for l2 in LANGUAGES:

                # Get the right place to look at for the data
                test_dataset = copy(dict_data[l2])
                if mode == 'full':
                    path_embed = f"models/classification_cnn/classification_CNN_{l1}_{epochs}e.pth"
                    # Even the char:int dictionaries
                    voc = copy(dict_models[l1]).word_voc_id
                    test_dataset.word_voc = copy(dict_models[l1]).word_voc
                    test_dataset.word_voc_id = voc
                else:
                    path_embed = f"models/classification_cnn/classification_CNN_{l2}_{epochs}e.pth"
                    # Even the char:int dictionaries
                    voc = copy(dict_models[l2]).word_voc_id
                    test_dataset.word_voc = copy(dict_models[l2]).word_voc
                    test_dataset.word_voc_id = voc

                # Prepare dataset
                BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
                EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value
                dataset_size = len(test_dataset)
                if dataset_size > nb_analogies:
                    indices = list(range(dataset_size))
                    sub_indices = rd.sample(indices, nb_analogies)
                    test_subset = Subset(test_dataset, sub_indices)
                    test_dataloader = DataLoader(test_subset, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))
                else:
                    test_dataloader = DataLoader(test_dataset, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))

                # Get models
                saved_classif = torch.load(path_classif)
                saved_embed = torch.load(path_embed)

                if (mode == 'full' and l1 == "japanese") or (mode == 'partial' and l2 == "japanese"):
                    emb_size = 512
                else:
                    emb_size = 64
                embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)
                embedding_model.load_state_dict(saved_embed['state_dict_embeddings'])
                embedding_model.eval()

                classification_model = Classification(emb_size=16*5)
                classification_model.load_state_dict(saved_classif['state_dict_classification'])
                classification_model.eval()

                embedding_model.to(device)
                classification_model.to(device)

                accuracy_raw = []
                accuracy_valid = []
                accuracy_invalid = []

                for a, b, c, d in test_dataloader:

                    # compute the embeddings
                    a = embedding_model(a.to(device))
                    b = embedding_model(b.to(device))
                    c = embedding_model(c.to(device))
                    d = embedding_model(d.to(device))

                    # Raw data

                    a2 = torch.unsqueeze(a, 0)
                    b2 = torch.unsqueeze(b, 0)
                    c2 = torch.unsqueeze(c, 0)
                    d2 = torch.unsqueeze(d, 0)

                    is_analogy = torch.squeeze(classification_model(a2, b2, c2, d2))
                    expected = torch.ones(is_analogy.size(), device=is_analogy.device)
                    accuracy_raw.append(get_accuracy_classification(expected, is_analogy))

                    # Valid and invalid data

                    data = torch.stack([a, b, c, d], dim = 1)

                    for a, b, c, d in enrich(data):

                        # valid examples, target is 1
                        a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                        b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                        c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                        d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                        is_analogy = torch.squeeze(classification_model(a, b, c, d))

                        expected = torch.ones(is_analogy.size(), device=is_analogy.device)

                        accuracy_valid.append(get_accuracy_classification(expected, is_analogy))

                    for a, b, c, d in generate_negative(data):

                        # invalid examples, target is 0
                        a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
                        b = torch.unsqueeze(torch.unsqueeze(b, 0), 0)
                        c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
                        d = torch.unsqueeze(torch.unsqueeze(d, 0), 0)

                        is_analogy = torch.squeeze(classification_model(a, b, c, d))

                        expected = torch.zeros(is_analogy.size(), device=is_analogy.device)

                        accuracy_invalid.append(get_accuracy_classification(expected, is_analogy))

                list_of_accuracies.append((mean(accuracy_valid), mean(accuracy_invalid), mean(accuracy_raw)))

                print(f"\tRun time: {elapsed():4.5}s, {l2} done, {list_of_accuracies[-1]}.")

            dic_of_accuracies[l1] = list_of_accuracies
            print(f"Run time: {elapsed():4.5}s, {l1} done.\n")

            # To store the results language per language
            #with open(f"all_accuracies-{l1}.csv", 'w') as f:
            #    f.write(str(dic_of_accuracies[l1]))

    # Store the full results : On a line the model used for the data of the columns
    df = pd.DataFrame(dic_of_accuracies, index = LANGUAGES).transpose()
    with open(f"{mode}_transfer.csv", 'w') as f:
        f.write(df.to_csv())

if __name__ == '__main__':
    evaluate_classifier_transfer()
