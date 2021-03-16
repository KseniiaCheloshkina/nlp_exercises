import torch
from torch import nn
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import copy
import razdel


def baseline_headline_generation(sents):
    try:
        sents[0]
    except:
        print(sents)
    return sents[0]


def calc_scores(references, predictions, metric="all"):
    print("Count:", len(predictions))
    print("Ref:", references[-1])
    print("Hyp:", predictions[-1])

    if metric in ("bleu", "all"):
        print("BLEU: ", corpus_bleu([[r] for r in references], predictions))
    if metric in ("rouge", "all"):
        rouge = Rouge()
        scores = rouge.get_scores(predictions, references, avg=True)
        print("ROUGE: ", scores)


class SentenceEncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, n_layers=3, dropout=0.3, bidirectional=True):
        super(SentenceEncoderRNN, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # TODO: initialize embeddings with smth
        self.embedding_layer = nn.Embedding(input_size, embedding_dim)
        self.rnn_layer = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs, hidden=None):
        embedded = self.embedding_layer(inputs)
        outputs, _ = self.rnn_layer(embedded, hidden)
        sentences_embeddings = torch.mean(outputs, 1)
        return sentences_embeddings


def build_oracle_summary_greedy(sentences, gold_summary, calc_score, lower=True):
    '''
    Жадное построение oracle summary
    '''
    gold_summary = gold_summary.lower() if lower else gold_summary
    # Делим текст на предложения
    n_sentences = len(sentences)
    oracle_summary_sentences = set()
    score = -1.0
    summaries = []
    try:
        for _ in range(n_sentences):
            for i in range(n_sentences):
                if i in oracle_summary_sentences:
                    continue
                current_summary_sentences = copy.copy(oracle_summary_sentences)
                # Добавляем какое-то предложения к уже существующему summary
                current_summary_sentences.add(i)
                current_summary = " ".join([sentences[index] for index in sorted(list(current_summary_sentences))])
                # Считаем метрики
                current_score = calc_score(current_summary, gold_summary)
                summaries.append((current_score, current_summary_sentences))
            # Если получилось улучшить метрики с добавлением какого-либо предложения, то пробуем добавить ещё
            # Иначе на этом заканчиваем
            best_summary_score, best_summary_sentences = max(summaries)
            if best_summary_score <= score:
                break
            oracle_summary_sentences = best_summary_sentences
            score = best_summary_score
        oracle_summary = " ".join([sentences[index] for index in sorted(list(oracle_summary_sentences))])
        return oracle_summary, oracle_summary_sentences
    except:
        print("calc_score exception")
        print(sentences)
        print(gold_summary)
        print(current_summary)
        print(current_summary_sentences)
        print(oracle_summary_sentences)
        return "", []


def calc_single_score(pred_summary, gold_summary, rouge):
    return rouge.get_scores([pred_summary], [gold_summary], avg=True)['rouge-2']['f']


def add_oracle_summary_to_records(records, max_sentences=30, lower=True, nrows=1000):
    rouge = Rouge()
    for i, record in enumerate(records):
        if i >= nrows:
            break
        # text = record["text"]
        sentences = record["sentences"]
        summary = record["title"]
        summary = summary.lower() if lower else summary
        sentences = sentences[:max_sentences]
        oracle_summary, sentences_indicies = build_oracle_summary_greedy(
            sentences,
            summary,
            calc_score=lambda x, y: calc_single_score(x, y, rouge),
            lower=lower
        )
        record["sentences"] = sentences
        record["oracle_sentences"] = list(sentences_indicies)
        record["oracle_summary"] = oracle_summary
    return records[:nrows]
