import stud.utils as utils

from typing import List, Dict
from model import Model
from stud.dataset import WiCDisambiguationDataset
from stud.classifier import WiCDisambiguationClassifier
from stud.word_embeddings import *


def build_model(device: str) -> Model:
    embedding_size = 200
    word_embeddings = GloVe(embedding_size=embedding_size)

    return StudentModel(device, word_embeddings)


class StudentModel(Model):

    def __init__(self, device: str, word_embeddings: WordEmbeddings, weights_path: str = 'model/weights.pt'):
        self.word_embeddings = word_embeddings
        self.model = WiCDisambiguationClassifier(word_embeddings.embedding_size, 100)
        self.model.load_weights(weights_path, device)

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        dataset = WiCDisambiguationDataset(sentence_pairs, self.word_embeddings, utils.sample2vector)

        result = self.model(torch.stack(dataset.encoded_samples))

        predictions = list()

        for i in range(len(sentence_pairs)):
            prediction = result['pred'][i].item()
            predictions.append('True' if prediction > 0.5 else 'False')

        return predictions
