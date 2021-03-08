from collections import Counter
import numpy as np
import torch
import torch.nn as nn


class Vocabulary(object):
    """
    Forms index2word, word2index and word2freq
    """
    def __init__(self):
        self.word2index = {
            "<unk>": 0
        }
        self.index2word = ["<unk>"]
        # frequency recording
        self.word2freq = {
            "<unk>": 0
        }

    def build(self, texts, min_count=10):
        words_counter = Counter(token for tokens in texts for token in tokens)
        for word, count in words_counter.most_common():
            if count >= min_count:
                self.word2index[word] = len(self.word2index)
                # frequency recording
                self.word2freq[word] = round(count ** 0.75)
        self.index2word = [word for word, _ in sorted(self.word2index.items(), key=lambda x: x[1])]
        # create freq list
        self.freq_table = []
        for word, count_df in self.word2freq.items():
            self.freq_table.extend([self.word2index[word]] * count_df)

    @property
    def size(self):
        return len(self.index2word)

    def top(self, n=100):
        return self.index2word[1:n + 1]

    def get_index(self, word):
        return self.word2index.get(word, 0)

    def get_word(self, index):
        return self.index2word[index]


def build_contexts(tokenized_texts, vocabulary, window_size):
    """ Returns List of (central_word, context)"""
    contexts = []
    for tokens in tokenized_texts:
        for i in range(len(tokens)):
            central_word = vocabulary.get_index(tokens[i])
            context = [vocabulary.get_index(tokens[i + delta]) for delta in range(-window_size, window_size + 1)
                       if delta != 0 and i + delta >= 0 and i + delta < len(tokens)]
            if len(context) != 2 * window_size:
                continue

            contexts.append((central_word, context))

    return contexts


class SkipGramModel(nn.Module):
    """
    Implements SkipGram model: predict each word in context by central word
    """

    def __init__(self, vocab_size, embedding_dim=32):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        projections = self.embeddings.forward(inputs)
        output = self.out_layer.forward(projections)
        return output

    @staticmethod
    def get_next_batch(contexts, window_size, batch_size, epochs_count):
        assert batch_size % (window_size * 2) == 0
        central_words, contexts = zip(*contexts)
        batch_size //= (window_size * 2)

        for epoch in range(epochs_count):
            indices = np.arange(len(contexts))
            np.random.shuffle(indices)
            batch_begin = 0
            while batch_begin < len(contexts):
                batch_indices = indices[batch_begin: batch_begin + batch_size]
                batch_contexts, batch_centrals = [], []
                for data_ind in batch_indices:
                    central_word, context = central_words[data_ind], contexts[data_ind]
                    batch_contexts.extend(context)
                    batch_centrals.extend([central_word] * len(context))

                batch_begin += batch_size
                yield torch.cuda.LongTensor(batch_centrals), torch.cuda.LongTensor(batch_contexts)


class CBOWModel(nn.Module):
    """
    Implements CBOW model: predict central words by weighted sum of words' embeddings in context
    """

    def __init__(self, vocab_size, embedding_dim=32):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        projections = self.embeddings.forward(inputs)
        sum_projections = torch.sum(projections, 1)
        output = self.out_layer.forward(sum_projections)
        return output

    @staticmethod
    def get_next_batch(contexts, window_size, batch_size, epochs_count):
        assert batch_size % (window_size * 2) == 0
        central_words, contexts = zip(*contexts)
        batch_size //= (window_size * 2)

        for epoch in range(epochs_count):
            indices = np.arange(len(contexts))
            np.random.shuffle(indices)
            batch_begin = 0
            while batch_begin < len(contexts):
                batch_indices = indices[batch_begin: batch_begin + batch_size]
                batch_contexts, batch_centrals = [], []
                for data_ind in batch_indices:
                    central_word, context = central_words[data_ind], contexts[data_ind]
                    batch_contexts.append(context)
                    batch_centrals.append(central_word)

                batch_begin += batch_size
                yield torch.cuda.LongTensor(batch_contexts), torch.cuda.LongTensor(batch_centrals)


class NegativeSamplingModel(nn.Module):
    """
    Implements NegativeSampling principle: predict whether each pair of words is in one context or not
    """
    def __init__(self, vocab_size, embedding_dim=32):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs):
        word_projections = self.word_embeddings.forward(inputs[:, 1])
        context_projections = self.context_embeddings.forward(inputs[:, 0])
        output = torch.diag(torch.tensordot(word_projections, torch.transpose(context_projections, 0, 1), dims=1))
        return output

    @staticmethod
    def get_next_batch(contexts, vocabulary, window_size, k_samples, batch_size, epochs_count):
        assert batch_size % (window_size * 2 + k_samples) == 0
        central_words, contexts = zip(*contexts)
        batch_size //= (window_size * 2 + k_samples)

        len_freq_list = len(vocabulary.freq_table)

        for epoch in range(epochs_count):
            indices = np.arange(len(contexts))
            np.random.shuffle(indices)
            batch_begin = 0
            while batch_begin < len(contexts):
                batch_indices = indices[batch_begin: batch_begin + batch_size]
                batch_contexts, batch_centrals, batch_targets = [], [], []
                for data_ind in batch_indices:
                    # positive examples
                    central_word, context = central_words[data_ind], contexts[data_ind]
                    batch_contexts.extend(context)
                    batch_centrals.extend([central_word] * len(context))
                    batch_targets.extend([1.0] * len(context))
                    # negative examples
                    indices_neg_current = np.random.randint(low=0, high=len_freq_list - 1, size=k_samples)
                    not_context = [vocabulary.freq_table[neg_data_ind] for neg_data_ind in indices_neg_current]
                    batch_contexts.extend(not_context)
                    batch_centrals.extend([central_word] * k_samples)
                    batch_targets.extend([0.0] * k_samples)
                batch_begin += batch_size

                inputs = torch.from_numpy(np.stack([batch_contexts, batch_centrals], axis=1)).cuda()
                yield torch.cuda.LongTensor(inputs), torch.cuda.LongTensor(batch_targets)