from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import nltk
import numpy as np

def tfidf_vectorizer(input_file):
    lemmatizer = WordNetLemmatizer()
    sw = stopwords.words('english')
    preprocessed_texts = []
    with open(input_file, 'r') as file:
        for text in file:
            words = word_tokenize(text)
            lemmas = [lemmatizer.lemmatize(word) for word in words]
            preprocessed_texts.append(' '.join(lemmas))

    tfidf_vec = TfidfVectorizer(stop_words=sw)
    tfidf_vec.fit(preprocessed_texts)
    return tfidf_vec

class TfIdfEmbedder(torch.nn.Module):
    def __init__(self, vectorizer):
        super(TfIdfEmbedder, self).__init__()
        self.vectorizer = vectorizer

    def forward(self, text):
        return self.vectorizer.transform([text]).toarray()