import stud.utils as utils

from typing import List, Dict, Set, Optional
from model import Model
from stud.dataset import WiCDisambiguationDataset
from stud.classifier import WiCDisambiguationClassifier
from stud.word_embeddings import *


def build_model(device: str) -> Model:
    word_embeddings = GloVe(embedding_size=200)
    weights_path = 'model/weights.pt'
    n_features = word_embeddings.embedding_size * 2
    hidden_size = n_features // 2
    stop_words = utils.load_stop_words('model/stop_words/english')

    return StudentModel(device, word_embeddings, weights_path, n_features, hidden_size, stop_words)


class StudentModel(Model):

    def __init__(self, device: str, word_embeddings: WordEmbeddings, weights_path: str, n_features: int,
                 hidden_size: int, stop_words: Optional[Set[str]] = None):
        self.word_embeddings = word_embeddings
        self.model = WiCDisambiguationClassifier(n_features, hidden_size)
        self.model.load_weights(weights_path, device)
        self.stop_words = stop_words

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        dataset = WiCDisambiguationDataset(sentence_pairs, self.word_embeddings, self.stop_words)

        result = self.model(torch.stack(dataset.encoded_samples))

        predictions = list()

        for i in range(len(sentence_pairs)):
            prediction = result['pred'][i].item()
            predictions.append('True' if prediction > 0.5 else 'False')

        return predictions
