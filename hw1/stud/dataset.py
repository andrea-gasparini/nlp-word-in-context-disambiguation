import torch
import jsonlines
import stud.utils as utils

from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Set
from stud.word_embeddings import WordEmbeddings


class WiCDisambiguationDataset(Dataset):
    target_word_weight = 10

    def __init__(self, samples: List[Dict], word_embeddings: WordEmbeddings, stop_words: Optional[Set[str]] = None) -> None:
        self.samples = samples
        self.word_embeddings = word_embeddings
        self.stop_words = stop_words

        self.encoded_samples, self.encoded_labels = self.encode_samples(self.samples)

    @staticmethod
    def from_file(path: str, word_embeddings: WordEmbeddings, stop_words: Optional[Set[str]] = None) -> 'WiCDisambiguationDataset':
        samples = list()

        with jsonlines.open(path) as f:
            for sample in f:
                samples.append(sample)

        return WiCDisambiguationDataset(samples, word_embeddings, stop_words)

    def encode_samples(self, samples: List[Dict]) -> Tuple[List[Tensor], List[Optional[Tensor]]]:
        encoded_samples = list()
        encoded_labels = list()

        for sample in samples:
            sentence1_vector = self.__sentence_to_vector(sample['sentence1'], int(sample['start1']))
            sentence2_vector = self.__sentence_to_vector(sample['sentence2'], int(sample['start2']))

            sentence1_mean = torch.mean(sentence1_vector, dim=0)
            sentence2_mean = torch.mean(sentence2_vector, dim=0)

            encoded_samples.append(torch.cat((sentence1_mean, sentence2_mean)))

            if 'label' in sample:
                encoded_label = torch.tensor(float(1 if utils.str_to_bool(sample['label']) else 0))
                encoded_labels.append(encoded_label)
            else:
                encoded_labels.append(None)

        return encoded_samples, encoded_labels

    def __sentence_to_vector(self, sentence: str, target_start: int) -> Tensor:
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

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx) -> Tuple[List[Tensor], List[Optional[Tensor]]]:
        return self.encoded_samples[idx], self.encoded_labels[idx]
