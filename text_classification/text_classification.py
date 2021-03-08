from collections import Counter
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy


class Vocabulary:
    def __init__(self):
        self.word2index = {
            "<pad>": 0,
            "<unk>": 1
        }
        self.index2word = ["<pad>", "<unk>"]

    def build(self, texts, min_count=7):
        words_counter = Counter(token for tokens in texts for token in tokens)
        for word, count in words_counter.most_common():
            if count >= min_count:
                self.word2index[word] = len(self.word2index)
        self.index2word = [word for word, _ in sorted(self.word2index.items(), key=lambda x: x[1])]

    def get_pad(self):
        return self.word2index["<pad>"]

    def get_unk(self):
        return self.word2index["<unk>"]

    @property
    def size(self):
        return len(self.index2word)

    def top(self, n=100):
        return self.index2word[1:n + 1]

    def get_index(self, word):
        return self.word2index.get(word, self.get_unk())

    def get_word(self, index):
        return self.index2word[index]


class SimpleModel(LightningModule):
    def __init__(self, vocab_size, embedding_dim=32):
        super().__init__()

        self.embeddings_layer = nn.Embedding(vocab_size, embedding_dim)
        self.loss = nn.BCEWithLogitsLoss()
        self.valid_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, inputs, labels):
        raise NotImplementedError("forward not implemented")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optimizer]

    def training_step(self, batch, _):
        inputs, labels = batch
        loss, logits = self(inputs, labels)
        return loss

    def validation_step(self, batch, _):
        inputs, labels = batch
        val_loss, logits = self(inputs, labels)
        self.valid_accuracy.update(logits, labels)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", self.valid_accuracy)

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.valid_accuracy.compute(), prog_bar=True)

    def test_step(self, batch, _):
        inputs, labels = batch
        test_loss, logits = self(inputs, labels)
        self.test_accuracy.update(logits, labels)
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy)

    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.test_accuracy.compute(), prog_bar=True)


class CharCnnModel(SimpleModel):
    """
    Run CNN with `filters` over symbols embeddings, take max pooling for each filter and add linear projection
    """
    def __init__(self, vocab_size, embedding_dim=128, filters=((2, 10), (3, 8))):
        super().__init__(vocab_size, embedding_dim)

        self.filters = []
        all_filters_count = 0
        for kernel_size, filters_count in filters:
            all_filters_count += filters_count
            self.filters.append(nn.Conv2d(1, filters_count, (kernel_size, embedding_dim), padding=(1, 0)))
        self.filters = nn.ModuleList(self.filters)
        self.relu_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout(0.2)
        self.out_layer = nn.Linear(all_filters_count, 1)

    def forward(self, inputs, labels):
        """
        Forward pass
        :param inputs: (batch_size, n_symbols_in_text_fixed_padded)
        text -> ["h", "e", "l", "l", "o", ..] -> [9, 56, 72, ..] + padding or cutting to max sequence length
        :param labels: (batch_size, 1)
        :return: loss and logits
        """
        projections = self.embeddings_layer.forward(inputs)
        # batch_size x num_words x embedding_dim
        projections = projections.unsqueeze(1)
        # batch_size x 1 x num_words x embedding_dim
        results = []
        for f in self.filters:
            convolved = self.dropout_layer(self.relu_layer(f(projections))).squeeze(3)
            # batch_size x num_filters x num_words
            pooling = torch.max(convolved, 2)[0]
            # batch_size x num_filters
            results.append(pooling)
        output = torch.cat(results, 1)
        # batch_size x sum(num_filters)
        logits = self.out_layer.forward(output)
        loss = self.loss(logits, labels)
        return loss, logits


class FastTextLSTMModel(LightningModule):
    """
    Run LSTM over tokens FastText embeddings and take final hidden state, add linear projection and dropout
    """
    def __init__(self, ft_embedding_dim, hidden_dim=64):
        super().__init__()

        self.lstm_layer = nn.LSTM(ft_embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(0.2)
        self.out_layer = nn.Linear(hidden_dim * 2, 1)

        self.loss = nn.BCEWithLogitsLoss()
        self.valid_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, embeddings, labels):
        """
        Forward pass
        :param embeddings: (batch_size, max_tokens_in_text, ft_embedding_dim)
        text -> ["hello", ",", "world",  ..] -> [9, 56, 72, ..] + padding or cutting to max sequence length
        :param labels: (batch_size, 1)
        :return: loss and logits
        """
        batch_size = embeddings.size(0)
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(embeddings)
        final_hidden_state = final_hidden_state.transpose(0, 1)
        final_hidden_state = final_hidden_state.reshape(batch_size, -1)
        text_hidden = self.dropout_layer(final_hidden_state)
        logits = self.out_layer.forward(text_hidden)
        loss = self.loss(logits, labels)
        return loss, logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optimizer]

    def training_step(self, batch, _):
        inputs, labels = batch
        loss, logits = self(inputs, labels)
        return loss

    def validation_step(self, batch, _):
        inputs, labels = batch
        val_loss, logits = self(inputs, labels)
        self.valid_accuracy.update(logits, labels)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", self.valid_accuracy)

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.valid_accuracy.compute(), prog_bar=True)

    def test_step(self, batch, _):
        inputs, labels = batch
        test_loss, logits = self(inputs, labels)
        self.test_accuracy.update(logits, labels)
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy)

    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.test_accuracy.compute(), prog_bar=True)
