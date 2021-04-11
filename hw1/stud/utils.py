import torch

from typing import Optional, Dict
from stud.word_embeddings import WordEmbeddings


def sample2vector(word_embeddings: WordEmbeddings, sample: Dict) -> Optional[torch.Tensor]:
    sentence_word_vector1 = [word_embeddings[w] for w in sample['sentence1'].split(' ') if w in word_embeddings]
    sentence_word_vector2 = [word_embeddings[w] for w in sample['sentence2'].split(' ') if w in word_embeddings]
    sentence_word_vector = sentence_word_vector1 + sentence_word_vector2

    if len(sentence_word_vector) == 0:
        return None

    sentence_word_vector = torch.stack(sentence_word_vector)

    return torch.mean(sentence_word_vector, dim=0)


def str_to_bool(s: str) -> bool:
    return s == 'True'
