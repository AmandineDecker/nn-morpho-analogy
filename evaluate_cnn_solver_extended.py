import click
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchtext.vocab as vocab
import random as rd
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import copy
from functools import partial
from statistics import mean, pstdev
from data import Task1Dataset, enrich
from analogy_reg import AnalogyRegression
from cnn_embeddings import CNNEmbedding
from utils import elapsed_timer, collate
from store_cnn_embeddings import generate_embeddings_file

def get_mins(t, k):
    '''Returns the indices of the tensors whose distance is close to the distance of the first of the list.

    Function to use with Euclidean distance.

    Arguments:
    t -- The tensor of distances.
    k -- The percentage of extra distance where to retrieve the vectors.'''

    minimums = {'bare': t.argmin().item(), 'mins_0': []}

    sort, indices = torch.sort(t)

    mins = []
    i = 0
    i_min = sort[0].item()

    for looseness in range(k+1):
        if looseness > 0:
            minimums[f'mins_{looseness}'] = minimums[f'mins_{looseness-1}'].copy()
        while sort[i].item() <= (i_min*(1+looseness/100)) and i < len(sort):
            minimums[f'mins_{looseness}'].append(indices[i].item())
            i += 1
    return minimums

def get_maxs(t, k):
    '''Returns the indices of the tensors whose distance is close to the distance of the first of the list.

    Function to use with Cosine similarity.

    Arguments:
    t -- The tensor of distances.
    k -- The percentage of extra distance where to retrieve the vectors.'''

    maximums = {'bare': t.argmax().item(), 'maxs_0': []}

    sort, indices = torch.sort(t, descending = True)

    maxs = []
    i = 0
    i_max = sort[0].item()

    for looseness in range(k+1):
        if looseness > 0:
            maximums[f'maxs_{looseness}'] = maximums[f'maxs_{looseness-1}'].copy()
        while sort[i].item() >= (i_max*(1-looseness/100)) and i < len(sort):
            maximums[f'maxs_{looseness}'].append(indices[i].item())
            i += 1
    return maximums

@click.command()
@click.option('--nb_analogies', default=50000, prompt='The number of analogies',
              help='The number of analogies (before augmentation) we evaluate the model on.', show_default=True)
@click.option('--epochs', default=20,
              help='The number of epochs the model was trained on (we use this parameter to use the right files).', show_default=True)
@click.option('--k', default=5, prompt='The range',
              help='The range we will search the right vector in.', show_default=True)
def evaluate_solver(nb_analogies, epochs, k):
    '''Produces the accuracy for all the analogy solvers.

    We look for the right vector in a given range around the closest to the produced one.

    Arguments:
    nb_analogies -- The maximum number of analogies (before augmentation) we evaluate the model on. If the number is greater than the number of analogies in the dataset, then all the analogies will be used.
    epochs -- The number of epochs the models were trained on (we use this parameter to use the right files).
    k -- The percentage of extra distance where to retrieve the vectors.'''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    LANGUAGES = ["arabic", "finnish", "georgian", "german", "hungarian", "japanese", "maltese", "navajo", "russian", "spanish", "turkish"]

    results = {}

    for language in LANGUAGES:

        custom_embeddings = vocab.Vectors(name = f'embeddings/char_cnn/{language}-vectors.txt',
                                          cache = 'embeddings/char_cnn',
                                          unk_init = torch.Tensor.normal_)

        custom_embeddings.vectors = custom_embeddings.vectors.to(device)

        # Cosine distance
        stored_lengths = torch.sqrt((custom_embeddings.vectors ** 2).sum(dim=1))

        def closest_cosine(vec, k):
            numerator = (custom_embeddings.vectors * vec).sum(dim=1)
            denominator = stored_lengths * torch.sqrt((vec ** 2).sum())
            similarities = numerator / denominator
            return get_maxs(similarities, k)

        # Euclidian distance
        def closest_euclid(vec, k):
            dists = torch.sqrt(((custom_embeddings.vectors - vec) ** 2).sum(dim=1))
            return get_mins(dists, k)

        def is_close_list(vecs, expected):
            result = False
            i = 0
            while (not result) and (i < len(vecs)):
                if torch.allclose(expected, custom_embeddings.vectors[vecs[i]].to(device), atol=1e-05):
                    result = True
                i += 1
            if result:
                return result, i
            else:
                return result, -1

        train_dataset = Task1Dataset(language=language, mode="train", word_encoding="char")
        if language == "japanese":
            japanese_train_analogies, japanese_test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = 42)

            test_dataset = copy(train_dataset)
            test_dataset.analogies = japanese_test_analogies

            train_dataset.analogies = japanese_train_analogies
        else:
            test_dataset = Task1Dataset(language=language, mode="test", word_encoding="char")
        voc = train_dataset.word_voc_id
        test_dataset.word_voc = train_dataset.word_voc
        test_dataset.word_voc_id = voc

        BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
        EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

        if len(test_dataset) > nb_analogies:
            test_indices = list(range(len(test_dataset)))
            rd.seed(BOS_ID)
            test_sub_indices = rd.sample(test_indices, nb_analogies)
            test_subset = Subset(test_dataset, test_sub_indices)
        else:
            test_subset = test_dataset

        # Load data
        test_dataloader = DataLoader(test_subset, shuffle = True, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))


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


        ######## Test models ########

        regression_model.to(device)
        embedding_model.to(device)

        list_keys = ['accuracy_cosine_', 'len_list_cosine_', 'id_right_tensor_cosine_', 'accuracy_euclid_', 'len_list_euclid_', 'id_right_tensor_euclid_']
        dico_result = {ke: [] for ke in [txt + str(looseness) for txt in list_keys for looseness in range(k+1)]}
        dico_result['accuracy_cosine_bare'] = []
        dico_result['accuracy_euclid_bare'] = []

        nb_analogies = 0

        with elapsed_timer() as elapsed:

            for a, b, c, d in test_dataloader:

                # compute the embeddings

                a = embedding_model(a.to(device))
                b = embedding_model(b.to(device))
                c = embedding_model(c.to(device))
                d = embedding_model(d.to(device))

                data = torch.stack([a, b, c, d], dim = 1).to(device)

                for a, b, c, d_expected in enrich(data):

                    nb_analogies += 1

                    d_pred = regression_model(a, b, c)

                    d_closest_cosine = closest_cosine(d_pred, k)
                    d_closest_euclid = closest_euclid(d_pred, k)

                    # bare results
                    dico_result['accuracy_cosine_bare'].append(torch.allclose(d_expected, custom_embeddings.vectors[d_closest_cosine['bare']], atol=1e-05))
                    dico_result['accuracy_euclid_bare'].append(torch.allclose(d_expected, custom_embeddings.vectors[d_closest_euclid['bare']], atol=1e-05))

                    # results for r
                    for looseness in range(k+1):
                        # cosine similarity
                        result_cosine, how_far_cosine = is_close_list(d_closest_cosine[f'maxs_{looseness}'], d_expected)
                        dico_result[f"len_list_cosine_{looseness}"].append(len(d_closest_cosine[f'maxs_{looseness}']))
                        dico_result[f"accuracy_cosine_{looseness}"].append(result_cosine)
                        dico_result[f"id_right_tensor_cosine_{looseness}"].append(how_far_cosine)
                        # euclidean distance
                        result_euclid, how_far_euclid = is_close_list(d_closest_euclid[f'mins_{looseness}'], d_expected)
                        dico_result[f"len_list_euclid_{looseness}"].append(len(d_closest_euclid[f'mins_{looseness}']))
                        dico_result[f"accuracy_euclid_{looseness}"].append(result_euclid)
                        dico_result[f"id_right_tensor_euclid_{looseness}"].append(how_far_euclid)

            results[language] = {"nb_analogies": nb_analogies,
                                "accuracy_cosine_bare": mean(dico_result['accuracy_cosine_bare']),
                                "accuracy_euclid_bare": mean(dico_result['accuracy_euclid_bare'])}
            for looseness in range(k+1):
                # cosine similarity - accuracy
                results[language][f"accuracy_cosine_{looseness}"] = mean(dico_result[f"accuracy_cosine_{looseness}"])
                # more than one tensor to check
                more_than_one_cosine = [x for x in dico_result[f"len_list_cosine_{looseness}"] if x != 1]
                results[language][f"more_than_one_cosine_{looseness}"] = len(more_than_one_cosine)
                results[language][f"mean_more_than_one_cosine_{looseness}"] = mean(more_than_one_cosine) if len(more_than_one_cosine) > 0  else -1
                results[language][f"sd_more_than_one_cosine_{looseness}"] = pstdev(more_than_one_cosine) if len(more_than_one_cosine) > 0  else -1
                # how far is the right tensor
                id_right_tensor_cosine = [x for x in dico_result[f"id_right_tensor_cosine_{looseness}"] if x > 1]
                id_right_tensor_found_cosine = [x for x in dico_result[f"id_right_tensor_cosine_{looseness}"] if x > 0]
                results[language][f"how_many_right_tensor_cosine_{looseness}"] = len(id_right_tensor_cosine)
                results[language][f"mean_id_right_tensor_cosine_{looseness}"] = mean(id_right_tensor_cosine) if len(id_right_tensor_cosine) > 0  else -1
                results[language][f"sd_id_right_tensor_cosine_{looseness}"] = pstdev(id_right_tensor_cosine) if len(id_right_tensor_cosine) > 0  else -1
                results[language][f"mrr_cosine_{looseness}"] = sum([1/x for x in id_right_tensor_found_cosine])/len(id_right_tensor_found_cosine) if len(id_right_tensor_found_cosine) > 0  else -1
                results[language][f"mrr_cosine_no_ones_{looseness}"] = sum([1/x for x in id_right_tensor_cosine])/len(id_right_tensor_cosine) if len(id_right_tensor_cosine) > 0  else -1
                results[language][f"list_of_ids_cosine_{looseness}"] = dico_result[f"id_right_tensor_cosine_{looseness}"]
                # euclid similarity - accuracy
                results[language][f"accuracy_euclid_{looseness}"] = mean(dico_result[f"accuracy_euclid_{looseness}"])
                # more than one tensor to check
                more_than_one_euclid = [x for x in dico_result[f"len_list_euclid_{looseness}"] if x != 1]
                results[language][f"more_than_one_euclid_{looseness}"] = len(more_than_one_euclid)
                results[language][f"mean_more_than_one_euclid_{looseness}"] = mean(more_than_one_euclid) if len(more_than_one_euclid) > 0  else -1
                results[language][f"sd_more_than_one_euclid_{looseness}"] = pstdev(more_than_one_euclid) if len(more_than_one_euclid) > 0  else -1
                # how far is the right tensor
                id_right_tensor_euclid = [x for x in dico_result[f"id_right_tensor_euclid_{looseness}"] if x > 1]
                id_right_tensor_found_euclid = [x for x in dico_result[f"id_right_tensor_euclid_{looseness}"] if x > 0]
                results[language][f"how_many_right_tensor_euclid_{looseness}"] = len(id_right_tensor_euclid)
                results[language][f"mean_id_right_tensor_euclid_{looseness}"] = mean(id_right_tensor_euclid) if len(id_right_tensor_euclid) > 0  else -1
                results[language][f"sd_id_right_tensor_euclid_{looseness}"] = pstdev(id_right_tensor_euclid) if len(id_right_tensor_euclid) > 0  else -1
                results[language][f"mrr_euclid_{looseness}"] = sum([1/x for x in id_right_tensor_found_euclid])/len(id_right_tensor_found_euclid) if len(id_right_tensor_found_euclid) > 0  else -1
                results[language][f"mrr_euclid_no_ones_{looseness}"] = sum([1/x for x in id_right_tensor_euclid])/len(id_right_tensor_euclid) if len(id_right_tensor_euclid) > 0  else -1
                results[language][f"list_of_ids_euclid_{looseness}"] = dico_result[f"id_right_tensor_euclid_{looseness}"]

            print(language, f' {elapsed():4.5}s')

    df = pd.DataFrame(results)
    with open(f"cnn_solver_{k}_extended_results.csv", 'w') as f:
        f.write(df.to_csv())

if __name__ == '__main__':
    evaluate_solver()
