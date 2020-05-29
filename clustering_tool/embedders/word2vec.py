import torch
import numpy as np
import spacy

def check_token(token):
    return not token.is_oov and not token.is_stop

class SpacyTokenizer:
    def __init__(self, spacyTokenizer):
        self.spacyTokenizer = spacyTokenizer

    def encode(self, text, **kwargs):
        return [token.text for token in self.spacyTokenizer(text) if check_token(token)]

def mean_embeddings(tokens, emb):
    return np.mean(emb, axis=-2)

def WeightedAverageEmbeddings(token2freq):
    def f(tokens, emb):
        weights = np.zeros(emb.shape)
        for i in range(emb.shape[0]):
            for j, token in enumerate(tokens[i]):
                if check_token(token) and token.text in token2freq:
                    weights[i, j] = 0.1 / (0.1 + token2freq[token.text])
        return np.average(emb, axis=-2, weights=weights)
    return f

def max_embeddings(tokens, emb):
    return np.max(emb, axis=-2)

class Word2VecEmbedder(torch.nn.Module):
    def __init__(self, spacyModelName, embedding_strategy = mean_embeddings):
        super(Word2VecEmbedder, self).__init__()
        self.spacyModel = spacy.load(spacyModelName)
        self.embedding_strategy = embedding_strategy

    def forward(self, text):
        encoded = self.spacyModel(text)
        tokens = [encoded]
        vectors = encoded.tensor[None, :, :]
        return self.embedding_strategy(tokens, vectors)