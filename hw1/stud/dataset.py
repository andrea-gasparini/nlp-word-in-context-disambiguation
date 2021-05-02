import torch
import stud.utils as utils

from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Set, Union
from stud.word_embeddings import WordEmbeddings
from abc import ABC as ABSTRACT_CLASS, abstractmethod


class WiCDisambiguationDataset(ABSTRACT_CLASS, Dataset):

    def __init__(self, samples: List[Dict], word_embeddings: WordEmbeddings, stop_words: Optional[Set[str]] = None):
        self.samples = samples
        self.word_embeddings = word_embeddings
        self.stop_words = stop_words

        self.encoded_samples, self.encoded_labels = self._encode_samples(self.samples)

    @staticmethod
    @abstractmethod
    def from_file(self, word_embeddings: WordEmbeddings, stop_words: Optional[Set[str]] = None) -> 'WiCDisambiguationDataset':
        pass

    def _encode_samples(self, samples: List[Dict]) -> Tuple[List[Union[Tensor, Tuple[Tensor, Tensor]]], List[Optional[Tensor]]]:
        encoded_samples = list()
        encoded_labels = list()

        for sample in samples:
            encoded_samples.append(self.encode_sample(sample))

            if 'label' in sample:
                encoded_label = torch.tensor(float(1 if utils.str_to_bool(sample['label']) else 0))
                encoded_labels.append(encoded_label)
            else:
                encoded_labels.append(None)

        return encoded_samples, encoded_labels

    @abstractmethod
    def encode_sample(self, sample: Dict) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        pass

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx) -> Tuple[Tensor, Optional[Tensor]]:
        return self.encoded_samples[idx], self.encoded_labels[idx]


class WordLevelWiCDisambiguationDataset(WiCDisambiguationDataset):
    target_word_weight = 10

    def __init__(self, samples: List[Dict], word_embeddings: WordEmbeddings, stop_words: Optional[Set[str]] = None):
        super().__init__(samples, word_embeddings, stop_words)

    @staticmethod
    def from_file(path: str, word_embeddings: WordEmbeddings, stop_words: Optional[Set[str]] = None) -> 'WordLevelWiCDisambiguationDataset':
        return WordLevelWiCDisambiguationDataset(utils.load_samples(path), word_embeddings, stop_words)

    def encode_sample(self, sample: Dict) -> Tensor:
        sentence1_vector = self._sentence_to_vector(sample['sentence1'], int(sample['start1']))
        sentence2_vector = self._sentence_to_vector(sample['sentence2'], int(sample['start2']))

        sentence1_mean = torch.mean(sentence1_vector, dim=0)
        sentence2_mean = torch.mean(sentence2_vector, dim=0)

        return torch.cat((sentence1_mean, sentence2_mean))

    def _sentence_to_vector(self, sentence: str, target_start: int) -> Tensor:
        sentence = utils.substitute_spacing_characters(sentence).lower()
        target_word_index = sentence[:target_start].count(' ')

        sentence_word_vector = list()

        for i, word in enumerate(sentence.split(' ')):
            if self.stop_words is None or word not in self.stop_words:
                word = utils.remove_special_characters(word)
                word_vector = self.word_embeddings[word]
                if i == target_word_index:
                    word_vector * self.target_word_weight
                sentence_word_vector.append(word_vector)

        return torch.stack(sentence_word_vector)


class LSTMWiCDisambiguationDataset(WiCDisambiguationDataset):

    def __init__(self, samples: List[Dict], word_embeddings: WordEmbeddings, stop_words: Optional[Set[str]] = None) -> None:
        super().__init__(samples, word_embeddings, stop_words)

    @staticmethod
    def from_file(path: str, word_embeddings: WordEmbeddings, stop_words: Optional[Set[str]] = None) -> 'LSTMWiCDisambiguationDataset':
        return LSTMWiCDisambiguationDataset(utils.load_samples(path), word_embeddings, stop_words)

    def encode_sample(self, sample: Dict) -> Tuple[Tensor, Tensor]:
        sentence1_indexes = torch.tensor(self.__sentence_to_indexes(sample['sentence1']))
        sentence2_indexes = torch.tensor(self.__sentence_to_indexes(sample['sentence2']))
        return sentence1_indexes, sentence2_indexes

    def __sentence_to_indexes(self, sentence: str) -> List[int]:
        sentence = utils.substitute_spacing_characters(sentence).lower()
        sentence_word_indexes = list()

        for i, word in enumerate(sentence.split(' ')):
            if self.stop_words is None or word not in self.stop_words:
                word = utils.remove_special_characters(word)
                word_indices = self.word_embeddings.word_indexes[word]
                sentence_word_indexes.append(word_indices)

        return sentence_word_indexes
