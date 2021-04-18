import os
from collections import defaultdict

import torch

from abc import ABC as ABSTRACT_CLASS, abstractmethod


class WordEmbeddings(ABSTRACT_CLASS):
    embeddings_dir = "model/word_embeddings/"
    unknown_embedding_files = {
        200: "unknown.200d.txt"
    }

    @abstractmethod
    def __init__(self, embedding_size, words_limit: int = 100_000):
        self.words_limit = words_limit
        self.embedding_size = embedding_size
        self._init_unknown_embedding()
        self.word_vectors = defaultdict(lambda: self.UNK)

    def _init_unknown_embedding(self):
        assert self.embedding_size in self.unknown_embedding_files, f"Unsupported embedding size: {self.embedding_size}"

        filename = self.unknown_embedding_files[self.embedding_size]
        file_path = os.path.join(self.embeddings_dir, filename)
        assert os.path.isfile(file_path), f"unknown word embedding {filename} not found in {self.embeddings_dir}"

        with open(file_path) as f:
            vector = f.readline().strip().split(' ')
            self.UNK = torch.tensor([float(c) for c in vector])

    def __contains__(self, word: str) -> bool:
        return word in self.word_vectors

    def __getitem__(self, word: str) -> torch.Tensor:
        return self.word_vectors[word]


class GloVe(WordEmbeddings):
    glove_dir = "model/word_embeddings/GloVe/"
    embedding_files = {
        50: "glove.6B.50d.txt",
        100: "glove.6B.100d.txt",
        200: "glove.6B.200d.txt",
        300: "glove.6B.300d.txt"
    }

    def __init__(self, words_limit: int = 100_000, embedding_size: int = 200):
        assert embedding_size in self.embedding_files, f"Unsupported embedding size: {embedding_size}"

        super().__init__(embedding_size, words_limit)
        self._init_data()

    def _init_data(self):
        filename = self.embedding_files[self.embedding_size]
        file_path = os.path.join(self.glove_dir, filename)
        assert os.path.isfile(file_path), f"GloVe embedding {filename} not found in {self.glove_dir}"

        with open(file_path) as f:
            for i, line in enumerate(f):

                if len(self.word_vectors) >= self.words_limit:
                    break

                word, *vector = line.strip().split(' ')
                vector = torch.tensor([float(c) for c in vector])

                self.word_vectors[word] = vector
