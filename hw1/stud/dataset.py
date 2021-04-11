import torch
import jsonlines
import stud.utils as utils

from torch import Tensor
from torch.utils.data import Dataset
from typing import Callable, List, Tuple, Dict, Optional
from stud.word_embeddings import WordEmbeddings


class WiCDisambiguationDataset(Dataset):

    def __init__(self, samples: List[Dict], word_embeddings: WordEmbeddings,
                 sample2vector: Callable[[WordEmbeddings, Dict], Optional[Tensor]]) -> None:
        self.samples = samples
        self.word_embeddings = word_embeddings
        self.sample2vector = sample2vector

        self.encoded_samples, self.encoded_labels = self.encode_samples(self.samples)

    @staticmethod
    def from_file(path: str, word_embeddings: WordEmbeddings,
                  sample2vector: Callable[[WordEmbeddings, Dict], Optional[Tensor]]) -> 'WiCDisambiguationDataset':
        samples = list()

        with jsonlines.open(path) as f:
            for sample in f:
                samples.append(sample)

        return WiCDisambiguationDataset(samples, word_embeddings, sample2vector)

    def encode_samples(self, samples: List[Dict]) -> Tuple[List[Tensor], List[Optional[Tensor]]]:
        encoded_samples = list()
        encoded_labels = list()

        for sample in samples:
            encoded_samples.append(self.sample2vector(self.word_embeddings, sample))

            if 'label' in sample:
                encoded_label = torch.tensor(float(1 if utils.str_to_bool(sample['label']) else 0))
                encoded_labels.append(encoded_label)
            else:
                encoded_labels.append(None)

        return encoded_samples, encoded_labels

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx) -> Tuple[Tensor, Optional[Tensor]]:
        return self.encoded_samples[idx], self.encoded_labels[idx]
