import os
import fasttext
import numpy as np
from razdel import tokenize
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from text_classification import FastTextLSTMModel, Vocabulary, download_dataset


class FastTextDataset(Dataset):
    def __init__(self, tokenized_texts, labels, ft_model, max_seq_len=50):
        self.ft_model = ft_model
        self.vector_dim = self.ft_model.get_dimension()
        self.samples = []
        assert len(tokenized_texts) != 0
        assert len(tokenized_texts) == len(labels)
        for i, (text, label) in enumerate(zip(tokenized_texts, labels)):
            text = text[:max_seq_len]
            embeddings = np.zeros((max_seq_len, self.vector_dim))
            embeddings[:len(text), :] = self.words_to_embeddings(text)
            self.samples.append((torch.FloatTensor(embeddings), torch.FloatTensor([label])))
            if i % 10000 == 0:
                print(i)

    def words_to_embeddings(self, words):
        vector_dim = self.ft_model.get_dimension()
        embeddings = np.zeros((len(words), vector_dim))
        for i, w in enumerate(words):
            embeddings[i] = self.ft_model.get_word_vector(w)
            embeddings[i] /= np.linalg.norm(embeddings[i])
        return embeddings

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def prepare_dataset(df_train, df_val, df_test):
    train_labels = df_train["label"].tolist()
    val_labels = df_val["label"].tolist()
    test_labels = df_test["label"].tolist()

    # text -> list of tokens
    train_texts = [[token.text for token in tokenize(text)] for text in df_train["text"].tolist()]
    val_texts = [[token.text for token in tokenize(text)] for text in df_val["text"].tolist()]
    test_texts = [[token.text for token in tokenize(text)] for text in df_test["text"].tolist()]

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def main():
    """
    Results in:
    Accuracy on train: 0.795
    Accuracy on val: 0.783
    Accuracy on test: 0.782
    :return:
    """
    df_train, df_val, df_test = download_dataset()
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = prepare_dataset(df_train,
                                                                                                      df_val,
                                                                                                      df_test)
    vocabulary = Vocabulary()
    vocabulary.build(train_texts)

    if "ft_ru_tweets_model_v2.bin" not in os.listdir():
        os.system("wget https://www.dropbox.com/s/t7fthf8axi30hct/ft_ru_tweets_model_v2.bin")
    ft_model = fasttext.load_model('ft_ru_tweets_model_v2.bin')
    BATCH_SIZE = 128

    ft_train_data = FastTextDataset(train_texts, train_labels, ft_model)
    ft_train_sampler = RandomSampler(ft_train_data)
    ft_train_loader = DataLoader(ft_train_data, batch_size=BATCH_SIZE, sampler=ft_train_sampler)

    ft_val_data = FastTextDataset(val_texts, val_labels, ft_model)
    ft_val_loader = DataLoader(ft_val_data, batch_size=BATCH_SIZE)

    ft_test_data = FastTextDataset(test_texts, test_labels, ft_model)
    ft_test_loader = DataLoader(ft_test_data, batch_size=BATCH_SIZE)

    ft_lstm_model = FastTextLSTMModel(128)
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
        max_epochs=20,
        progress_bar_refresh_rate=10,
        callbacks=[early_stop_callback])
    trainer.fit(ft_lstm_model, ft_train_loader, ft_val_loader)
    print("Testing on TRAIN:")
    trainer.test(ft_lstm_model, ft_train_loader)
    print("Testing on VAL:")
    trainer.test(ft_lstm_model, ft_val_loader)
    print("Testing on TEST:")
    trainer.test(ft_lstm_model, ft_test_loader)


if __name__ == "__main__":
    main()
