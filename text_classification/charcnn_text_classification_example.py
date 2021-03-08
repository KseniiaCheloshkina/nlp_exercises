import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from text_clf.text_classification import CharCnnModel, Vocabulary


class CharsDataset(Dataset):
    def __init__(self, tokenized_texts, labels, vocabulary, max_seq_len=200):
        self.samples = []
        assert len(tokenized_texts) != 0
        assert len(tokenized_texts) == len(labels)
        for text, label in zip(tokenized_texts, labels):
            indices = [vocabulary.get_index(word) for word in text][:max_seq_len]
            indices += [vocabulary.get_pad() for _ in range(max_seq_len - len(indices))]
            self.samples.append((torch.LongTensor(indices), torch.FloatTensor([label])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def download_dataset():
    # download
    os.system("wget https://www.dropbox.com/s/fnpq3z4bcnoktiv/positive.csv")
    os.system("wget https://www.dropbox.com/s/r6u59ljhhjdg6j0/negative.csv")

    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
    data_positive = pd.read_csv('positive.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])
    data_negative = pd.read_csv('negative.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])

    sample_size = min(data_positive.shape[0], data_negative.shape[0])
    raw_data = np.concatenate((data_positive['text'].values[:sample_size], data_negative['text'].values[:sample_size]),
                              axis=0)

    def preprocess_text(text):
        text = text.lower().replace("ё", "е")
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
        text = re.sub('@[^\s]+', 'USER', text)
        text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    df_train = pd.DataFrame(columns=['text', 'label'])
    df_test = pd.DataFrame(columns=['text', 'label'])

    data = [preprocess_text(t) for t in raw_data]
    labels = [1] * sample_size + [0] * sample_size
    df_train['text'], df_test['text'], df_train['label'], df_test['label'] = train_test_split(data, labels,
                                                                                              test_size=0.2,
                                                                                              random_state=1)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=1)
    return df_train, df_val, df_test


def prepare_dataset(df_train, df_val, df_test):

    train_labels = df_train["label"].tolist()
    val_labels = df_val["label"].tolist()
    test_labels = df_test["label"].tolist()

    # text -> list of symbols
    train_texts = [list(text) for text in df_train["text"].tolist()]
    val_texts = [list(text) for text in df_val["text"].tolist()]
    test_texts = [list(text) for text in df_test["text"].tolist()]
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def main():
    df_train, df_val, df_test = download_dataset()
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = prepare_dataset(
        df_train, df_val, df_test)

    vocabulary = Vocabulary()
    vocabulary.build(train_texts)

    BATCH_SIZE = 128

    train_data = CharsDataset(train_texts, train_labels, vocabulary)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)

    val_data = CharsDataset(val_texts, val_labels, vocabulary)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    test_data = CharsDataset(test_texts, test_labels, vocabulary)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    cnn_model = CharCnnModel(vocabulary.size, embedding_dim=50, filters=((15,15), (10, 10), (8,8), (6, 6)))
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=1,
        verbose=True,
        mode="min"
    )
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=False,
        accumulate_grad_batches=1,
        max_epochs=10,
        progress_bar_refresh_rate=10,
        callbacks=[early_stop_callback])
    trainer.fit(cnn_model, train_loader, val_loader)
    trainer.test(cnn_model, test_loader)
