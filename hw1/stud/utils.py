import torch
import re

from typing import Optional, List, Set, Tuple
from torch import Tensor


def rnn_collate_fn(data_elements: List[Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]]) \
        -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Optional[Tensor]]:
    X1 = [de[0][0] for de in data_elements]
    X2 = [de[0][1] for de in data_elements]

    X1_lengths = torch.tensor([x.size(0) for x in X1], dtype=torch.long)
    X2_lengths = torch.tensor([x.size(0) for x in X2], dtype=torch.long)

    X = X1 + X2
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)

    Y = [de[1] for de in data_elements if de[1] is not None]
    Y = torch.tensor(Y) if len(Y) == len(data_elements) else None

    return (X[0:len(X1)], X[len(X1):len(X)]), (X1_lengths, X2_lengths), Y


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
