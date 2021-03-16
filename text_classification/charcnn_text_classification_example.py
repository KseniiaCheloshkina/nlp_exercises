import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from text_classification import CharCnnModel, Vocabulary, download_dataset


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
    """
    Results in:
    Accuracy on train: 0.761
    Accuracy on val: 0.738
    Accuracy on test: 0.738
    :return:
    """
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
        max_epochs=20,
        progress_bar_refresh_rate=10,
        callbacks=[early_stop_callback])
    trainer.fit(cnn_model, train_loader, val_loader)
    print("Testing on TRAIN:")
    trainer.test(cnn_model, train_loader)
    print("Testing on VAL:")
    trainer.test(cnn_model, val_loader)
    print("Testing on TEST:")
    trainer.test(cnn_model, test_loader)


if __name__ == "__main__":
    main()
