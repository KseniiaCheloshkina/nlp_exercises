import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np


class CharCNNWordLstmModel(nn.Module):
    def __init__(self, char_set_size, char_embedding_dim=4, classes_count=3, word_embedding_dim=16, lstm_embedding_dim=16, char_max_seq_len=40, filters=((2, 10), (3, 8))):
        super().__init__()
        
        self.embeddings_layer = nn.Embedding(char_set_size, char_embedding_dim)

        self.filters = []
        all_filters_count = 0
        for kernel_size, filters_count in filters:
            all_filters_count += filters_count
            self.filters.append(nn.Conv2d(1, filters_count, (kernel_size, char_embedding_dim), padding=(1, 0)))
        self.filters = nn.ModuleList(self.filters)
        self.relu_layer = nn.ReLU()
        self.cnn_dropout_layer = nn.Dropout(0.2)

        self.lstm_layer = nn.LSTM(all_filters_count, lstm_embedding_dim // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.out_layer = nn.Linear(lstm_embedding_dim, classes_count)

    def forward(self, inputs):
        # inputs (batch_size, max_tokens_in_text, max_chars_in_token)
        projections = self.embeddings_layer.forward(inputs)
        # projections (batch_size, max_tokens_in_text, max_chars_in_token, char_embedding_dim)
        batch_size = projections.size(0)
        words_len = projections.size(1)
        projections = projections.reshape(batch_size * words_len, projections.size(2), projections.size(3))
        # projections (batch_size * max_tokens_in_text, max_chars_in_token, char_embedding_dim)
        projections = projections.unsqueeze(1)
        # projections (batch_size * max_tokens_in_text, 1, max_chars_in_token, char_embedding_dim)
        results = []
        for f in self.filters:
            convolved = self.cnn_dropout_layer(self.relu_layer(f(projections)))
            convolved = convolved.squeeze(3)
            # convolved (batch_size * max_tokens_in_text, filters_count, cnn_out)
            pooling = torch.max(convolved, 2)[0]
            # pooling (batch_size * max_tokens_in_text, filters_count)
            results.append(pooling)
        output = torch.cat(results, 1)
        # output (batch_size * max_tokens_in_text, all_filters_count)
        output = output.reshape(batch_size, words_len, output.size(1))
        # projections (batch_size, max_tokens_in_text, all_filters_count)
        output, _= self.lstm_layer(output)
        # output (batch_size, max_tokens_in_text, lstm_embedding_dim)
        output = self.dropout(output)
        output = self.out_layer.forward(output)
        # output (batch_size, max_tokens_in_text, lstm_embedding_dim)
        return output

def get_spans(labels, tokens):
    spans = []
    for i, label in enumerate(labels):
        if label == 1:
            spans.append((tokens[i].start, tokens[i].stop, "PER"))
        elif label == 2:
            spans[-1] = (spans[-1][0], tokens[i].stop, spans[-1][-1])
    return spans


def calc_metrics(true_labels, predicted_labels, samples):
    one_tp = 0
    one_fp = 0
    one_fn = 0
    for true, predicted in zip(true_labels, predicted_labels):
        for l1, l2 in zip(true, predicted):
            if l1 == 1 and l2 == 1:
                one_tp += 1
            elif l1 != 1 and l2 == 1:
                one_fp += 1
            elif l1 == 1 and l2 !=1:
                one_fn += 1
    if one_tp + one_fp == 0:
        print("No positives!")
    else:
        print("1 Precision: {}, 1 Recall: {}".format(float(one_tp)/(one_tp + one_fp), float(one_tp)/(one_tp + one_fn)))

    exact = 0
    partial = 0
    missing = 0
    spurius = 0
    for (true, predicted), sample in zip(zip(true_labels, predicted_labels), samples):
        true_spans = get_spans(true, sample.tokens)
        predicted_spans = get_spans(predicted, sample.tokens)
        for true_span in true_spans:
            is_missing = True
            for predicted_span in predicted_spans:
                if true_span == predicted_span:
                    exact += 1
                    is_missing = False
                    break
                ts = true_span[0]
                te = true_span[1]
                ps = predicted_span[0]
                pe = predicted_span[1]
                # ts te ps pe
                # ps pe ts te
                if ts <= te <= ps <= pe or ps <= pe <= ts <= te:
                    continue
                is_missing = False
                partial += 1
                break
            if is_missing:
                missing += 1
        for predicted_span in predicted_spans:
            is_missing = True
            for true_span in true_spans:
                if true_span == predicted_span:
                    is_missing = False
                    break
                ts = true_span[0]
                te = true_span[1]
                ps = predicted_span[0]
                pe = predicted_span[1]
                if ts <= te <= ps <= pe or ps <= pe <= ts <= te:
                    continue
                is_missing = False
                break
            if is_missing:
                spurius += 1
    print("Exact: {}, partial: {}, missing: {}, spurius: {}".format(exact, partial, missing, spurius))
            


def predict(model, samples):
    model.eval()
    true_labels = []
    predicted_labels = []
    all_indices = []
    for _, (indices, batch, batch_labels) in enumerate(get_next_gen_batch(samples)):
        logits = model(batch)
        plabels = logits.max(dim=2)[1]
        # Убираем неконсистентность
        for sample_num, sample in enumerate(plabels):
            for word_num, label in enumerate(sample):
                if label != 2:
                    continue
                if word_num == 0:
                    plabels[sample_num][word_num] = 0
                    continue
                if sample[word_num - 1] != 1:
                    plabels[sample_num][word_num] = 0
        true_labels.extend(batch_labels)
        predicted_labels.extend(plabels)
        all_indices.extend(indices)
    samples = [samples[index] for index in all_indices]
    calc_metrics(true_labels, predicted_labels, samples)
    show_box_markup(samples[0].text, get_spans(predicted_labels[0], samples[0].tokens), palette=palette(PER=BLUE, ORG=RED, LOC=GREEN))


def get_next_gen_batch(samples, max_seq_len=500, max_char_seq_len=40, batch_size=32):
    indices = np.arange(len(samples))
    np.random.shuffle(indices)
    batch_begin = 0
    while batch_begin < len(samples):
        batch_indices = indices[batch_begin: batch_begin + batch_size]
        batch = []
        batch_labels = []
        batch_max_len = 0
        for data_ind in batch_indices:
            sample = samples[data_ind]
            # список из токенов текста, каждый токен - список индексов символов
            inputs = []
            for token in sample.tokens[:max_seq_len]:
                # каждое слово как список индексов символов макс длины max_char_seq_len
                chars = [char_set.index(ch) if ch in char_set else char_set.index("<unk>") for ch in token.text][:max_char_seq_len]
                # добиваем каждое слово до длины max_char_seq_len
                chars += [0] * (max_char_seq_len - len(chars))
                inputs.append(chars)
            batch_max_len = max(batch_max_len, len(inputs))
            # добиваем каждый текст до длины max_seq_len
            inputs += [[0]*max_char_seq_len] * (max_seq_len - len(inputs))
            batch.append(inputs)
            # добиваем метки до длины max_seq_len
            labels = sample.labels[:max_seq_len]
            labels += [0] * (max_seq_len - len(labels))
            batch_labels.append(labels)
        batch_begin += batch_size
        batch = torch.cuda.LongTensor(batch)[:, :batch_max_len]
        labels = torch.cuda.LongTensor(batch_labels)[:, :batch_max_len]
        yield batch_indices, batch, labels


def train_gen_model(model, train_samples, val_samples, epochs_count=10, 
                    loss_every_nsteps=1000, lr=0.01, save_path="model.pt", device_name="cuda",
                    early_stopping=True):
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: {}".format(params_count))
    device = torch.device(device_name)
    model = model.to(device)
    total_loss = 0
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss().cuda()
    prev_avg_val_loss = None
    for epoch in range(epochs_count):
        model.train()
        for step, (_, batch, batch_labels) in enumerate(get_next_gen_batch(train)):
            logits = model(batch) # Прямой проход
            # (batch_size, max_tokens_in_text, classes_count)
            logits = logits.transpose(1, 2)
            # (batch_size, classes_count, max_tokens_in_text)
            loss = loss_function(logits, batch_labels) # Подсчёт ошибки batch_labels (batch_size, max_tokens_in_text)
            loss.backward() # Подсчёт градиентов dL/dw
            optimizer.step() # Градиентный спуск или его модификации (в данном случае Adam)
            optimizer.zero_grad() # Зануление градиентов, чтобы их спокойно менять на следующей итерации
            total_loss += loss.item()
        val_total_loss = 0
        val_batch_count = 0
        model.eval()
        for _, (_, batch, batch_labels) in enumerate(get_next_gen_batch(val)):
            logits = model(batch) # Прямой проход
            logits = logits.transpose(1, 2)
            val_total_loss += loss_function(logits, batch_labels) # Подсчёт ошибки
            val_batch_count += 1
        avg_val_loss = val_total_loss/val_batch_count
        print("Epoch = {}, Avg Train Loss = {:.4f}, Avg val loss = {:.4f}, Time = {:.2f}s".format(epoch, total_loss / loss_every_nsteps, avg_val_loss, time.time() - start_time))
        total_loss = 0
        start_time = time.time()

        if early_stopping and prev_avg_val_loss is not None and avg_val_loss > prev_avg_val_loss:
            model.load_state_dict(torch.load(save_path))
            model.eval()
            break
        prev_avg_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)


