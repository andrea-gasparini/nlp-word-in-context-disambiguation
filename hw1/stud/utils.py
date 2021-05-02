import re

from typing import Set


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
