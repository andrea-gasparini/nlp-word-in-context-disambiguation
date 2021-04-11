import torch
import re

from typing import Optional, Dict
from stud.word_embeddings import WordEmbeddings


def sample2vector(word_embeddings: WordEmbeddings, sample: Dict) -> Optional[torch.Tensor]:
    sentence1 = remove_special_characters(sample['sentence1']).lower()
    sentence2 = remove_special_characters(sample['sentence2']).lower()
    sentences_concat = sentence1 + " | " + sentence2

    sentences_word_vector = [word_embeddings[word] for word in sentences_concat.split(' ')]

    if len(sentences_word_vector) == 0:
        return None

    sentences_word_vector = torch.stack(sentences_word_vector)

    return torch.mean(sentences_word_vector, dim=0)


def remove_special_characters(string: str) -> str:
    string = re.sub(r'[\-—]', ' ', string)
    string = re.sub(r'[.,:;|!?@#$"“”()\[\]&\\<>/0-9]', '', string)

    return string


def str_to_bool(s: str) -> bool:
    return s == 'True'
