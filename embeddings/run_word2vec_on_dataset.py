import os
import time
import re
import datetime as dt
from string import punctuation
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from razdel import tokenize, sentenize

import word2vec
from word2vec import Vocabulary, build_contexts


def prepare_dataset():
    # download
    os.system("wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz")
    os.system("gzip -d lenta-ru-news.csv.gz")
    os.system("head -n 2 lenta-ru-news.csv")

    def get_date(url):
        dates = re.findall(r"\d\d\d\d\/\d\d\/\d\d", url)
        return next(iter(dates), None)

    def get_texts(dataset):
        texts = []
        for text in dataset["text"]:
            for sentence in sentenize(text):
                texts.append([token.text.lower() for token in tokenize(sentence.text) if token.text not in punctuation])

        for title in dataset["title"]:
            texts.append([token.text.lower() for token in tokenize(title) if token.text not in punctuation])
        return texts

    # read and transform to a list of words
    dataset = pd.read_csv("lenta-ru-news.csv", sep=',', quotechar='\"', escapechar='\\', encoding='utf-8', header=0)
    dataset["date"] = dataset["url"].apply(lambda x: dt.datetime.strptime(get_date(x), "%Y/%m/%d"))
    dataset = dataset[dataset["date"] > "2017-01-01"]
    dataset["text"] = dataset["text"].apply(lambda x: x.replace("\xa0", " "))
    dataset["title"] = dataset["title"].apply(lambda x: x.replace("\xa0", " "))
    train_dataset = dataset[dataset["date"] < "2018-04-01"]
    test_dataset = dataset[dataset["date"] > "2018-04-01"]
    texts = get_texts(train_dataset)
    test_texts = get_texts(test_dataset)

    return texts, test_texts


def main(method="skigram"):
    train_texts, test_texts = prepare_dataset()

    vocabulary = Vocabulary()
    vocabulary.build(train_texts)
    assert vocabulary.word2index[vocabulary.index2word[10]] == 10
    print(vocabulary.size)
    print(vocabulary.top(100))

    contexts = build_contexts(train_texts, vocabulary, window_size=2)
    print(contexts[:5])
    print(vocabulary.get_word(contexts[0][0]), [vocabulary.get_word(index) for index in contexts[0][1]])

    # TRAIN MODEL
    model = train_w2vec(vocabulary, contexts, method)

    # final embeddings
    embeddings = model.embeddings.weight.cpu().data.numpy()
    np.save("embeddings.npy", embeddings)


def train_w2vec(vocabulary, contexts, method):
    if method == "skigram":
        algo = "SkipGramModel"
    elif method == "cbow":
        algo = "CBOWModel"
    else:
        raise NotImplementedError("This method ({}) is not implemented. Should be on of 'skipgram'/'cbow'".format(method))
    model_name = getattr(word2vec, algo)
    model = model_name(vocabulary.size, 32)

    device = torch.device("cuda")
    model = model.to(device)

    loss_every_nsteps = 1000
    total_loss = 0
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss().cuda()

    for step, (batch_inputs, batch_outputs) in enumerate(
            model.get_next_batch(contexts, window_size=2, batch_size=256, epochs_count=10)):
        logits = model(batch_inputs)
        loss = loss_function(logits, batch_outputs.type_as(logits))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if step != 0 and step % loss_every_nsteps == 0:
            print("Step = {}, Avg Loss = {:.4f}, Time = {:.2f}s".format(step, total_loss / loss_every_nsteps,
                                                                        time.time() - start_time))
            total_loss = 0
            start_time = time.time()

    return model
