import torch
import re

from typing import Optional, List, Set
from stud.word_embeddings import WordEmbeddings


def sample2vector(word_vectors: WordEmbeddings, sample: dict, separator: str = '|', target_weight: int = 10) -> torch.Tensor:
    sentence1 = sentence2embeddings(word_vectors, sample['sentence1'], int(sample['start1']), target_weight)
    sentence2 = sentence2embeddings(word_vectors, sample['sentence2'], int(sample['start2']), target_weight)

    sentences_word_vector = sentence1 + [word_vectors[separator]] + sentence2
    sentences_word_vector = torch.stack(sentences_word_vector)

    return torch.mean(sentences_word_vector, dim=0)


def sentence2embeddings(word_vectors: WordEmbeddings, sentence: str, target_start: int, target_weight: int,
                        stop_words: Optional[Set[str]] = None) -> List[torch.Tensor]:
    sentence = substitute_spacing_characters(sentence).lower()
    target_word_index = sentence[:target_start].count(' ')

    sentence_word_vector = list()

    for i, word in enumerate(sentence.split(' ')):
        if stop_words is None or word not in stop_words:
            word = remove_special_characters(word)
            word_vector = word_vectors[word]
            if i == target_word_index:
                word_vector * target_weight
            sentence_word_vector.append(word_vector)

    return sentence_word_vector


def substitute_spacing_characters(string: str) -> str:
    return re.sub(r'[\-—_/]', ' ', string)


def remove_special_characters(string: str) -> str:
    return re.sub(r'[.,:;|!?@#$"“”\'’()\[\]&\\<>0-9]', '', string)


def str_to_bool(s: str) -> bool:
    return s == 'True'


def load_stop_words(path: str) -> Set[str]:
    stop_words = set()
    with open(path) as f:
        for line in f:
            stop_words.add(line.strip())

    return stop_words
