from data import Task1Dataset
import torch, torch.nn as nn
from cnn_embeddings import CNNEmbedding
import torch.nn.functional as F
import numpy
from utils import pad
from sklearn.model_selection import train_test_split
from copy import copy

def encode_word(voc, word):
    '''Encodes a word into a list of IDs thanks to a character to integer mapping.

    Arguments:
    voc -- The character to integer dictionary.
    word -- The word to encode.'''
    return [voc[c] if c in voc.keys() else 0 for c in word]

def generate_embeddings_file(language, path_embed):
    '''Stores the embeddings of the training and test set of a given language and returns the path to the file.

    Arguments:
    language -- Language of the words to store.
    path_embed -- The path to the embedding model to use.'''

    train_dataset = Task1Dataset(language=language, mode="train", word_encoding="char")
    if language == "japanese":
        japanese_train_analogies, japanese_test_analogies = train_test_split(train_dataset.analogies, test_size=0.3, random_state = 42)

        test_dataset = copy(train_dataset)
        test_dataset.analogies = japanese_test_analogies

        train_dataset.analogies = japanese_train_analogies
    else:
        test_dataset = Task1Dataset(language=language, mode="test", word_encoding="char")
    voc = train_dataset.word_voc_id
    test_dataset.word_voc_id = voc
    test_dataset.word_voc = train_dataset.word_voc

    train_dic = {word: encode_word(train_dataset.word_voc_id, word) for word in train_dataset.all_words}
    test_dic = {word: encode_word(test_dataset.word_voc_id, word) for word in test_dataset.all_words}
    train_dic.update(test_dic)
    vocabulary = train_dic.copy()

    BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
    EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

    if language == "japanese":
        emb_size = 512
    else:
        emb_size = 64

    saved_embed = torch.load(path_embed)
    embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)
    embedding_model.load_state_dict(saved_embed['state_dict_embeddings'])
    embedding_model.eval()


    with open(f"embeddings/char_cnn/{language}-vectors.txt", 'w') as f:
        for word, embed in vocabulary.items():
            embedding = torch.unsqueeze(torch.LongTensor(embed), 0)
            embedding = embedding_model(pad(embedding, BOS_ID, EOS_ID))
            embedding = torch.squeeze(embedding)
            embedding = embedding.tolist()
            embedding = [str(i) for i in embedding]
            embedding = ' '.join(embedding)
            f.write(f"{word} {embedding}\n")

    return f"embeddings/char_cnn/{language}-vectors.txt"
