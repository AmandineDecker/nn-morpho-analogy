import torch, torch.nn as nn
import random

PATH = "./sigmorphon2016/data/"
LANGUAGES = ["arabic", "finnish", "georgian", "german", "hungarian", "japanese", "maltese", "navajo", "russian", "spanish", "turkish"]
LANGS = ["ar", "fi", "ka", "de", "hu", "mt", "nv", "ru", "es", "tr"] # for BPEmb subword embeddings
MODES = ["train", "dev", "test", "test-covered"]

def load_data(language="german", mode="train", task=2):
    '''Load the data from the sigmorphon files in the form of a list of triples (lemma, target_features, target_word).'''
    assert language in LANGUAGES, f"Language '{language}' is unkown, allowed languages are {LANGUAGES}"
    assert mode in MODES, f"Mode '{mode}' is unkown, allowed modes are {MODES}"
    if language == 'japanese':
        assert mode == 'train', f"Mode '{mode}' is unkown for Japanese, the only allowed mode is 'train'"
    filename = f"{language}-task{task}-{mode}"
    with open(PATH + filename, "r", encoding="utf-8") as f:
        return [line.strip().split('\t') for line in f]

def enrich(data):
    """Apply the example generation process from 'Solving Word Analogies: a Machine Learning Perspective'."""
    for a, b, c, d in data:
        yield a, b, c, d
        yield c, d, a, b
        yield c, a, d, b
        yield d, b, c, a
        yield d, c, b, a
        yield b, a, d, c
        yield b, d, a, c
        yield a, c, b, d

def generate_negative(positive_data):
    """Apply the negative example generation process from 'Solving Word Analogies: a Machine Learning Perspective'."""
    for a, b, c, d in positive_data:
        yield b, a, c, d
        yield c, b, a, d
        yield a, a, c, d

class Task2Dataset(torch.utils.data.Dataset):
    def __init__(self, language="german", mode="train", feature_encoding = "char", word_encoding="none"):
        super(Task2Dataset).__init__()
        self.language = language
        self.mode = mode
        self.feature_encoding = feature_encoding
        self.word_encoding = word_encoding
        self.raw_data = load_data(language = language, mode = mode, task = 2)

        self.prepare_data()

    def prepare_data(self):
        """Generate embeddings for the 4 elements.

        There are 3 modes to encode the features:
        - 'feature-value': sequence of each indivitual feature, wrt. a dictioanry of values;
        - 'sum': the one-hot vectors derived using 'feature-value' are summed, resulting in a vector of dimension corresponding to the number of possible values for all the possible features;
        - 'char': sequence of ids of characters, wrt. a dictioanry of values.

        There are 2 modes to encode the words:
        - 'glove': [only for German] pre-trained GloVe embedding of the word;
        - 'char': sequence of ids of characters, wrt. a dictioanry of values;
        - 'none' or None: no encoding, particularly useful when coupled with BERT encodings.
        """
        if self.feature_encoding == "char":
            # generate character vocabulary
            voc = set()
            for feature_a, word_a, feature_b, word_b in self.raw_data:
                voc.update(feature_a)
                voc.update(feature_b)
            self.feature_voc = list(voc)
            self.feature_voc.sort()
            self.feature_voc_id = {character: i for i, character in enumerate(self.feature_voc)}

        elif self.feature_encoding == "feature-value" or self.feature_encoding == "sum":
            # generate feature-value vocabulary
            voc = set()
            for feature_a, word_a, feature_b, word_b in self.raw_data:
                voc.update(feature_a.split(","))
                voc.update(feature_b.split(","))
            self.feature_voc = list(voc)
            self.feature_voc.sort()
            self.feature_voc_id = {character: i for i, character in enumerate(self.feature_voc)}
        else:
            print(f"Unsupported feature encoding: {self.feature_encoding}")


        if self.word_encoding == "char":
            # generate character vocabulary
            voc = set()
            for feature_a, word_a, feature_b, word_b in self.raw_data:
                voc.update(word_a)
                voc.update(word_b)
            self.word_voc = list(voc)
            self.word_voc.sort()
            self.word_voc_id = {character: i for i, character in enumerate(self.word_voc)}

        elif self.word_encoding == "glove":
            from embeddings.glove import GloVe
            self.glove = GloVe()

        elif self.word_encoding == "none" or self.word_encoding is None:
            pass

        else:
            print(f"Unsupported word encoding: {self.word_encoding}")

    def encode_word(self, word):
        if self.word_encoding == "char":
            return torch.LongTensor([self.word_voc_id[c] for c in word])
        elif self.word_encoding == "glove":
            return self.glove.embeddings.get(word, torch.zeros(300))
        elif self.word_encoding == "none" or self.word_encoding is None:
            return word
        else:
            raise ValueError(f"Unsupported word encoding: {self.word_encoding}")
    def encode_feature(self, feature):
        if self.feature_encoding == "char":
            return torch.LongTensor([self.feature_voc_id[c] for c in feature])
        elif self.feature_encoding == "feature-value" or self.feature_encoding == "sum":
            feature_enc = torch.LongTensor([self.feature_voc_id[feature] for feature in feature.split(",")])
            if self.feature_encoding == "sum":
                feature_enc = nn.functional.one_hot(feature_enc, num_classes=len(self.feature_voc_id)).sum(dim=0)
            return feature_enc
        else:
            raise ValueError(f"Unsupported feature encoding: {self.feature_encoding}")
    def encode(self, feature_a, word_a, feature_b, word_b):
        return self.encode_feature(feature_a), self.encode_word(word_a), self.encode_feature(feature_b), self.encode_word(word_b)

    def decode_word(self, word):
        if self.word_encoding == "char":
            return "".join([self.word_voc[char.item()] for char in word])
        elif self.word_encoding == "glove":
            print("Word decoding not supported with GloVe.")
        elif self.word_encoding == "none" or self.word_encoding is None:
            print("Word decoding not necessary when using 'none' encoding.")
            return word
        else:
            print(f"Unsupported word encoding: {self.word_encoding}")

    def decode_feature(self, feature):
        if self.word_encoding == "char":
            return "".join([self.feature_voc[char.item()] for char in feature])
        elif self.feature_encoding == "feature-value":
            return "".join([self.feature_voc[f.item()] for f in feature])
        elif self.feature_encoding == "sum":
            print("Feature decoding not supported with 'sum' encoding.")
        else:
            print(f"Unsupported feature encoding: {self.feature_encoding}")

    def __len__(self): return len(self.raw_data)
    def __getitem__(self, index): return self.encode(*self.raw_data[index])
    def words(self):
        for feature_a, word_a, feature_b, word_b in self:
            yield word_a
            yield word_b
    def features(self):
        for feature_a, word_a, feature_b, word_b in self:
            yield feature_a
            yield feature_b

class Task1Dataset(torch.utils.data.Dataset):

    def __init__(self, language="german", mode="train", word_encoding="none"):
        super(Task2Dataset).__init__()
        self.language = language
        self.mode = mode
        self.word_encoding = word_encoding
        self.raw_data = load_data(language = language, mode = mode, task=1)

        self.prepare_data()
        self.set_analogy_classes()

    def prepare_data(self):
        """Generate embeddings for the 4 elements.

        There are 2 modes to encode the words:
        - 'glove': [only for German] pre-trained GloVe embedding of the word;
        - 'char': sequence of ids of characters, wrt. a dictioanry of values;
        - 'none' or None: no encoding, particularly useful when coupled with BERT encodings.
        """
        if self.word_encoding == "char":
            # generate character vocabulary
            voc = set()
            for word_a, feature_b, word_b in self.raw_data:
                voc.update(word_a)
                voc.update(word_b)
            self.word_voc = list(voc)
            self.word_voc.sort()
            self.word_voc_id = {character: i for i, character in enumerate(self.word_voc)}

        elif self.word_encoding == "glove":
            from embeddings.glove import GloVe
            self.glove = GloVe()

        elif self.word_encoding == "none" or self.word_encoding is None:
            pass

        else:
            print(f"Unsupported word encoding: {self.word_encoding}")

    def set_analogy_classes(self):
        self.analogies = []
        self.all_words = set()
        for i, (word_a_i, feature_b_i, word_b_i) in enumerate(self.raw_data):
            self.all_words.add(word_a_i)
            self.all_words.add(word_b_i)
            for j, (word_a_j, feature_b_j, word_b_j) in enumerate(self.raw_data[i:]):
                if feature_b_i == feature_b_j:
                    self.analogies.append((i,i+j))
    def encode_word(self, word):
        if self.word_encoding == "char":
            return torch.LongTensor([self.word_voc_id[c] if c in self.word_voc_id.keys() else -1 for c in word])
            #return torch.LongTensor([self.word_voc_id[c] if c in self.word_voc_id.keys() else random.choice(list(self.word_voc_id.values())) for c in word])
        elif self.word_encoding == "glove":
            return self.glove.embeddings.get(word, torch.zeros(300))
        elif self.word_encoding == "none" or self.word_encoding is None:
            return word
        else:
            raise ValueError(f"Unsupported word encoding: {self.word_encoding}")
    def encode(self, a, b, c, d):
        return self.encode_word(a), self.encode_word(b), self.encode_word(c), self.encode_word(d)

    def decode_word(self, word):
        if self.word_encoding == "char":
            return "".join([self.word_voc[char.item()] for char in word])
        elif self.word_encoding == "glove":
            print("Word decoding not supported with GloVe.")
        elif self.word_encoding == "none" or self.word_encoding is None:
            print("Word decoding not necessary when using 'none' encoding.")
            return word
        else:
            print(f"Unsupported word encoding: {self.word_encoding}")

    def __len__(self):
        return len(self.analogies)
    def __getitem__(self, index):
        ab_index, cd_index = self.analogies[index]
        a, feature_b, b = self.raw_data[ab_index]
        c, feature_d, d = self.raw_data[cd_index]
        return self.encode(a, b, c, d)

if __name__ == "__main__":
    print(len(Task1Dataset().analogies))
    print(Task1Dataset()[2500])
    print(len(Task2Dataset()))
    print(Task2Dataset()[2500])
    print(Task2Dataset().raw_data[2500])
