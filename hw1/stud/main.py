import torch

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from stud.trainer import Trainer
from stud.dataset import WiCDisambiguationDataset
from stud.classifier import WiCDisambiguationClassifier, HParams
from stud.word_embeddings import GloVe
from stud.utils import bilstm_collate_fn, lstm_collate_fn


if __name__ == '__main__':
    word_embeddings = GloVe(embedding_size=200)
    stop_words = None # load_stop_words('model/stop_words/english')

    TRAIN_DATASET = WiCDisambiguationDataset.from_file("data/train.jsonl", word_embeddings, stop_words)
    DEV_DATASET = WiCDisambiguationDataset.from_file("data/dev.jsonl", word_embeddings, stop_words)

    hparams = HParams()
    hparams.lstm_bidirectional = True

    model = WiCDisambiguationClassifier(HParams(), word_embeddings)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if True else torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    trainer = Trainer(model, optimizer)
    model.save_weights("model/model_local_rnn.pt")

    logs = trainer.train(DataLoader(TRAIN_DATASET, batch_size=32, collate_fn=bilstm_collate_fn),
                         DataLoader(DEV_DATASET, batch_size=32, collate_fn=bilstm_collate_fn),
                         epochs=100, early_stopping=True, early_stopping_patience=3)

    print(f"Final training loss     => {logs['train_history'][-1]}")
    print(f"Final validation loss   => {logs['valid_history'][-1]}")
    trainer.plot_losses()

    labels = DEV_DATASET.encoded_labels
    predictions = list()

    for x, x_length, y in DataLoader(DEV_DATASET, batch_size=32, collate_fn=bilstm_collate_fn):
        prediction = model(x, x_length, y)['pred']
        predictions += [x.clone().detach() for x in torch.round(prediction)]

    p = precision_score(labels, predictions, average='macro')
    r = recall_score(labels, predictions, average='macro')
    f = f1_score(labels, predictions, average='macro')
    a = accuracy_score(labels, predictions)

    print(f'# precision: {p:.4f}')
    print(f'# recall: {r:.4f}')
    print(f'# f1: {f:.4f}')
    print(f'# acc: {a:.4f}')
