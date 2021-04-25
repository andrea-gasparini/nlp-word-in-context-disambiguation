import torch
import jsonlines
import stud.utils as utils

from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Set
from stud.word_embeddings import WordEmbeddings


class WiCDisambiguationDataset(Dataset):

    def __init__(self, samples: List[Dict], word_embeddings: WordEmbeddings, stop_words: Optional[Set[str]] = None) -> None:
        self.samples = samples
        self.stop_words = stop_words
        self.word_embeddings = word_embeddings

        self.encoded_samples, self.encoded_labels = self.encode_samples(self.samples)

    @staticmethod
    def from_file(path: str, word_embeddings: WordEmbeddings, stop_words: Optional[Set[str]] = None) -> 'WiCDisambiguationDataset':
        samples = list()

        with jsonlines.open(path) as f:
            for sample in f:
                samples.append(sample)

        return WiCDisambiguationDataset(samples, word_embeddings, stop_words)

    def encode_samples(self, samples: List[Dict]) -> Tuple[List[Tuple[Tensor, Tensor]], List[Optional[Tensor]]]:
        encoded_samples = list()
        encoded_labels = list()

        for sample in samples:
            sentence1_indexes = torch.tensor(self.__sentence_to_indexes(sample['sentence1']))
            sentence2_indexes = torch.tensor(self.__sentence_to_indexes(sample['sentence2']))
            encoded_samples.append((sentence1_indexes, sentence2_indexes))

            if 'label' in sample:
                encoded_label = torch.tensor(float(1 if utils.str_to_bool(sample['label']) else 0))
                encoded_labels.append(encoded_label)
            else:
                encoded_labels.append(None)

        return encoded_samples, encoded_labels

    def __sentence_to_indexes(self, sentence: str) -> List[int]:
        sentence = utils.substitute_spacing_characters(sentence).lower()
        sentence_word_indexes = list()

        for i, word in enumerate(sentence.split(' ')):
            if self.stop_words is None or word not in self.stop_words:
                word = utils.remove_special_characters(word)
                word_indices = self.word_embeddings.word_indexes[word]
                sentence_word_indexes.append(word_indices)

        return sentence_word_indexes

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx) -> Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]:
        return self.encoded_samples[idx], self.encoded_labels[idx]
