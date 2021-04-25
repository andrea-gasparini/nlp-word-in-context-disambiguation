import stud.utils as utils

from typing import List, Dict, Optional, Set
from model import Model
from torch.utils.data import DataLoader
from stud.dataset import WiCDisambiguationDataset
from stud.classifier import WiCDisambiguationClassifier, HParams
from stud.word_embeddings import *


def build_model(device: str) -> Model:
    word_embeddings = GloVe(embedding_size=200)
    weights_path = 'model/weights.pt'
    stop_words = utils.load_stop_words('model/stop_words/english')

    return StudentModel(device, word_embeddings, weights_path, stop_words)


class StudentModel(Model):

    def __init__(self, device: str, word_embeddings: WordEmbeddings, weights_path: str,
                 stop_words: Optional[Set[str]] = None) -> None:
        self.word_embeddings = word_embeddings
        self.model = WiCDisambiguationClassifier(HParams(), word_embeddings)
        self.model.load_weights(weights_path, device)
        self.stop_words = stop_words 

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        dataset = WiCDisambiguationDataset(sentence_pairs, self.word_embeddings, self.stop_words)
        predictions = list()

        for x, x_length, y in DataLoader(dataset, batch_size=32, collate_fn=utils.rnn_collate_fn):
            predictions_batch = self.model(x, x_length, y)['pred']
            predictions += ['True' if prediction > 0.5 else 'False' for prediction in predictions_batch]

        return predictions
