import stud.utils as utils

from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Set
from model import Model
from stud.dataset import WordLevelWiCDisambiguationDataset, LSTMWiCDisambiguationDataset
from stud.classifier import HParams, WordLevelWiCDisambiguationClassifier, LSTMWiCDisambiguationClassifier
from stud.word_embeddings import *


def build_model(device: str) -> Model:
    word_embeddings = GloVe(embedding_size=200)
    stop_words = utils.load_stop_words('model/stop_words/english')

    model = 'word-level'  # 'word-level' || 'lstm' || 'bi-lstm

    if model == 'word-level':
        weights_path = 'model/word-level-approach-weights.pt'
        n_features = word_embeddings.embedding_size * 2
        hidden_size = n_features // 2

        return WordLevelModel(device, n_features, hidden_size, word_embeddings, weights_path, stop_words)

    elif model == 'lstm' or model == 'bi-lstm':
        weights_path = 'model/lstm-approach-weights.pt'
        hparams = HParams()

        if model == 'bi-lstm':
            hparams.lstm_bidirectional = True
            stop_words = None

        return LSTMModel(device, hparams, word_embeddings, weights_path, stop_words)


class WordLevelModel(Model):

    def __init__(self, device: str, n_features: int, hidden_size: int, word_embeddings: WordEmbeddings,
                 weights_path: str, stop_words: Optional[Set[str]] = None):
        self.word_embeddings = word_embeddings
        self.model = WordLevelWiCDisambiguationClassifier(n_features, hidden_size)
        self.model.load_weights(weights_path, device)
        self.stop_words = stop_words

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        dataset = WordLevelWiCDisambiguationDataset(sentence_pairs, self.word_embeddings, self.stop_words)
        result = self.model(torch.stack(dataset.encoded_samples))
        predictions = list()

        for i in range(len(sentence_pairs)):
            prediction = result['pred'][i].item()
            predictions.append('True' if prediction > 0.5 else 'False')

        return predictions


class LSTMModel(Model):

    def __init__(self, device: str, hparams: HParams, word_embeddings: WordEmbeddings, weights_path: str,
                 stop_words: Optional[Set[str]] = None) -> None:
        self.word_embeddings = word_embeddings
        self.collate_fn = utils.bilstm_collate_fn if hparams.lstm_bidirectional else utils.lstm_collate_fn
        self.model = LSTMWiCDisambiguationClassifier(hparams, word_embeddings)
        self.model.load_weights(weights_path, device)
        self.stop_words = stop_words

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        dataset = LSTMWiCDisambiguationDataset(sentence_pairs, self.word_embeddings, self.stop_words)
        predictions = list()

        for x, x_length, y in DataLoader(dataset, batch_size=32, collate_fn=self.collate_fn):
            predictions_batch = self.model(x, x_length, y)['pred']
            predictions += ['True' if prediction > 0.5 else 'False' for prediction in predictions_batch]

        return predictions
