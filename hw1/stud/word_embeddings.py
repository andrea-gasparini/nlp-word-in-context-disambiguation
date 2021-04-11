import os
from collections import defaultdict

import torch

from abc import ABC as ABSTRACT_CLASS, abstractmethod


class WordEmbeddings(ABSTRACT_CLASS):

    @abstractmethod
    def __init__(self, embedding_size, words_limit: int = 100_000):
        self.words_limit = words_limit
        self.embedding_size = embedding_size
        self.UNK = torch.rand(embedding_size)
        self.word_vectors = defaultdict(lambda: self.UNK)

    def __contains__(self, word: str) -> bool:
        return word in self.word_vectors

    def __getitem__(self, word: str) -> torch.Tensor:
        return self.word_vectors[word]


class GloVe(WordEmbeddings):
    glove_dir = "hw1/stud/word_embeddings/GloVe/"
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
