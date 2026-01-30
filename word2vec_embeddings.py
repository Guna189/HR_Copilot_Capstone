from langchain.embeddings.base import Embeddings
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np


class Word2VecEmbeddings(Embeddings):
    def __init__(self, model_path):
        self.model = Word2Vec.load(model_path)
        self.vector_size = self.model.vector_size

    def _embed_text(self, text: str):
        tokens = word_tokenize(text.lower())
        vectors = [
            self.model.wv[token]
            for token in tokens
            if token in self.model.wv
        ]

        if not vectors:
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)

    def embed_documents(self, texts):
        return [self._embed_text(text).tolist() for text in texts]

    def embed_query(self, text):
        return self._embed_text(text).tolist()
