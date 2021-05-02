import torch

from torch import Tensor
from typing import Optional, Dict, Tuple
from abc import ABC as ABSTRACT_CLASS, abstractmethod
from stud.word_embeddings import WordEmbeddings


class WiCDisambiguationClassifier(ABSTRACT_CLASS, torch.nn.Module):

    @abstractmethod
    def __init__(self):
        super().__init__()

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: str) -> None:
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.eval()

    def loss(self, pred: Tensor, y: Tensor) -> Tensor:
        return self.loss_fn(pred, y)


class WordLevelWiCDisambiguationClassifier(WiCDisambiguationClassifier):

    def __init__(self, n_features: int, n_hidden: int) -> None:
        super().__init__()

        self.lin1 = torch.nn.Linear(n_features, n_hidden)
        self.lin2 = torch.nn.Linear(n_hidden, 1)
        self.relu = torch.nn.ReLU()
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Dict[str, Tensor]:
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out).squeeze(1)
        out = torch.sigmoid(out)

        result = {'pred': out}

        if y is not None:
            loss = self.loss(out, y)
            result['loss'] = loss

        return result


class HParams:
    hidden_dim = 100
    lstm_bidirectional = False
    lstm_layers = 1
    dropout = 0.5


class LSTMWiCDisambiguationClassifier(WiCDisambiguationClassifier):

    def __init__(self, hparams: HParams, word_embeddings: WordEmbeddings) -> None:
        super().__init__()

        self.vectors_store = word_embeddings.vectors_store

        # embedding layer
        self.embedding = torch.nn.Embedding.from_pretrained(self.vectors_store)

        # recurrent layer
        self.rnn = torch.nn.LSTM(input_size=self.vectors_store.size(1),
                                 hidden_size=hparams.hidden_dim,
                                 num_layers=hparams.lstm_layers,
                                 bidirectional=hparams.lstm_bidirectional,
                                 dropout=hparams.dropout if hparams.lstm_layers > 1 else 0,
                                 batch_first=True)

        lstm_output_dim = hparams.hidden_dim if hparams.lstm_bidirectional is False else hparams.hidden_dim * 2

        concat_sentences_dim = lstm_output_dim * 2

        # classification head
        self.lin1 = torch.nn.Linear(concat_sentences_dim, concat_sentences_dim)
        self.lin2 = torch.nn.Linear(concat_sentences_dim, 1)

        # regularization
        self.dropout = torch.nn.Dropout(hparams.dropout)
        self.relu = torch.nn.ReLU()

        # loss function
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, x: Tuple[Tensor, Tensor], x_length: Tuple[Tensor, Tensor], y: Optional[Tensor] = None) -> Dict[str, Tensor]:

        # embedding words from indices
        embedding_out_1 = self.__embedding_step(x[0])
        embedding_out_2 = self.__embedding_step(x[1])

        # recurrent encoding
        recurrent_out_1 = self.__recurrent_step(embedding_out_1)
        recurrent_out_2 = self.__recurrent_step(embedding_out_2)

        summary_vectors_1 = self.__get_summary_vectors(recurrent_out_1, x_length[0])
        summary_vectors_2 = self.__get_summary_vectors(recurrent_out_2, x_length[1])

        summary_vectors = torch.cat((summary_vectors_1, summary_vectors_2), dim=1)

        out = self.lin1(summary_vectors)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.lin2(out).squeeze(1)
        out = torch.sigmoid(out)

        result = {'pred': out}

        if y is not None:
            loss = self.loss(out, y)
            result['loss'] = loss

        return result

    def __embedding_step(self, x: Tensor) -> Tensor:
        embedding_out = self.embedding(x)
        embedding_out = self.dropout(embedding_out)

        return embedding_out

    def __recurrent_step(self, embedding_out: Tensor):
        recurrent_out, _ = self.rnn(embedding_out)
        recurrent_out = self.dropout(recurrent_out)

        return recurrent_out

    def __get_summary_vectors(self, recurrent_out: Tensor, x_length: Tensor):
        batch_size, seq_len, hidden_size = recurrent_out.shape

        # flattening the recurrent output to have a long sequence of (batch_size x seq_len) vectors
        flattened_out = recurrent_out.reshape(-1, hidden_size)

        # computing a tensor of the indices of the last token in each batch element
        last_word_relative_indices = x_length - 1
        sequences_offsets = torch.arange(batch_size) * seq_len
        summary_vectors_indices = sequences_offsets + last_word_relative_indices

        # summary vectors that should summarize the elements in the batch
        summary_vectors = flattened_out[summary_vectors_indices]

        return summary_vectors
