import torch
import re
import jsonlines

from typing import Optional, List, Set, Tuple, Dict
from torch import Tensor


def bilstm_collate_fn(data_elements: List[Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]]) \
        -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Optional[Tensor]]:
    return rnn_collate_fn(data_elements, target_word=True)


def lstm_collate_fn(data_elements: List[Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]]) \
        -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Optional[Tensor]]:
    return rnn_collate_fn(data_elements, target_word=False)


def rnn_collate_fn(data_elements: List[Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]], target_word: bool = False) \
        -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Optional[Tensor]]:
    X1 = [de[0][0]['sentence_word_indexes'] for de in data_elements]
    X2 = [de[0][1]['sentence_word_indexes'] for de in data_elements]

    X1_summary_position = None
    X2_summary_position = None

    if target_word:
        X1_summary_position = torch.tensor([de[0][0]['target_word_index'] for de in data_elements])
        X2_summary_position = torch.tensor([de[0][1]['target_word_index'] for de in data_elements])
    else:
        X1_summary_position = torch.tensor([x.size(0) - 1 for x in X1], dtype=torch.long)
        X2_summary_position = torch.tensor([x.size(0) - 1 for x in X2], dtype=torch.long)

    X = X1 + X2
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)

    Y = [de[1] for de in data_elements if de[1] is not None]
    Y = torch.tensor(Y) if len(Y) == len(data_elements) else None

    return (X[0:len(X1)], X[len(X1):len(X)]), (X1_summary_position, X2_summary_position), Y


def substitute_spacing_characters(string: str) -> str:
    """
    Returns a new string with spacing characters replaced by an actual space.

    Ex: 'word-in-context is difficult' --> 'word in context is difficult'
    """
    return re.sub(r'[\-—_/]', ' ', string)


def remove_special_characters(string: str) -> str:
    """
    Returns a new string without punctuation and special characters
    """
    return re.sub(r'[.,:;|!?@#$"“”\'’()\[\]&\\<>0-9]', '', string)


def str_to_bool(s: str) -> bool:
    return s == 'True'


def load_stop_words(path: str) -> Set[str]:
    stop_words = set()
    with open(path) as f:
        for line in f:
            stop_words.add(line.strip())

    return stop_words


def load_samples(path: str) -> List[Dict]:
    samples = list()

    with jsonlines.open(path) as f:
        for sample in f:
            samples.append(sample)

    return samples
