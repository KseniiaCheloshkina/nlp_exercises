# https://github.com/huggingface/transformers/blob/e6cff60b4cbc1158fbd6e4a1c3afda8dc224f566/examples/run_glue.py#L69
# https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=Ykk0P9JiKtVe
import random
import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from text_classification import download_dataset


def train():
    df_train, df_val, df_test = download_dataset()
    # Load the BERT tokenizer
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    BATCH_SIZE = 32

    # get dataloaders
    train_dataloader, validation_dataloader, test_dataloader = load_and_format(
        df_train[:100], df_val[:100], df_test[:100], tokenizer, BATCH_SIZE)

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Batch size: 16, 32
    # Learning rate (Adam): 5e-5, 3e-5, 2e-5
    # Number of epochs: 2, 3, 4
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5,
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    epochs = 1
    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    device = torch.device("cuda")

    model, df_stats = fit_model(
        epochs,
        model,
        train_dataloader,
        validation_dataloader,
        device,
        optimizer,
        scheduler
    )
    print(df_stats)
    save_model(model, tokenizer, output_dir="fine-tuned-bert")

    # evaluate
    # evaluate
    predictions, true_labels = predict(model, train_dataloader, device)
    print("Accuracy on train: {}".format(flat_accuracy(predictions[0], true_labels[0])))
    predictions, true_labels = predict(model, validation_dataloader, device)
    print("Accuracy on val: {}".format(flat_accuracy(predictions[0], true_labels[0])))
    predictions, true_labels = predict(model, test_dataloader, device)
    print("Accuracy on test: {}".format(flat_accuracy(predictions[0], true_labels[0])))
    return df_stats


def fit_model(
        epochs,
        model,
        train_dataloader,
        validation_dataloader,
        device,
        optimizer,
        scheduler,
):
    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. `dropout` and `batchnorm` layers behave differently during training vs. test
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = time.time() - t0
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a backward pass.
            # PyTorch doesn't do this because accumulating the gradients is "convenient while training RNNs".
            model.zero_grad()

            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

            loss = result.loss
            logits = result.logits

            # Accumulate the training loss over all of the batches
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = time.time() - t0

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        print("")
        print("Running Validation...")
        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)

            # Get the loss and "logits" output by the model. The "logits" are the
            # output values prior to applying an activation function like the
            # softmax.
            loss = result.loss
            logits = result.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = time.time() - t0

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(time.time() - total_t0))
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    return model, df_stats


def predict(model, dataloader, device):
    predictions, true_labels = [], []
    # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
    model.eval()

    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

        # Get the loss and "logits" output by the model. The "logits" are the
        # output values prior to applying an activation function like the
        # softmax.
        loss = result.loss
        logits = result.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    return predictions, true_labels


def save_model(model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # how to load afterwards
    # model = <model_class>.from_pretrained(output_dir)
    # tokenizer = <tokenizer_class>.from_pretrained(output_dir)
    # if device is not None:
    #     model.to(device)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_dataloader(sentences, labels, tokenizer, batch_size):
    input_ids, attention_masks = transform_input(sentences, tokenizer)
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    return dataloader


def load_and_format(df_train, df_val, df_test, tokenizer, batch_size):
    train_dataloader = get_dataloader(sentences=df_train["text"].tolist(),
                                      labels=df_train["label"].tolist(),
                                      tokenizer=tokenizer,
                                      batch_size=batch_size
                                      )
    validation_dataloader = get_dataloader(sentences=df_val["text"].tolist(),
                                      labels=df_val["label"].tolist(),
                                      tokenizer=tokenizer,
                                      batch_size=batch_size
                                      )
    test_dataloader = get_dataloader(sentences=df_test["text"].tolist(),
                                      labels=df_test["label"].tolist(),
                                      tokenizer=tokenizer,
                                      batch_size=batch_size
                                     )
    return train_dataloader, validation_dataloader, test_dataloader


def transform_input(sentences, tokenizer):
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []
    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=122,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def find_optimal_max_sentence_len(sentences, tokenizer):
    """
    Training time depends on max seq len so it should be selected carefully.
    """
    max_len = []
    # For every sentence...
    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len.append(len(input_ids))
    max_len = np.array(max_len)
    print('Max sentence length: ', np.max(max_len))
    print('95% sentence length: ', np.quantile(max_len, 0.95))
    print('99% sentence length: ', np.quantile(max_len, 0.99))


def eda_dataset_bert():
    df_train, df_val, df_test = download_dataset()
    sentences = df_train["text"].tolist()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    find_optimal_max_sentence_len(sentences, tokenizer)


def main():
    """
    Results in:
    Accuracy on train: 0.90
    Accuracy on val: 0.80
    Accuracy on test: 0.78
    :return:
    """
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    df_stats = train()
    print(df_stats)
    return df_stats


if __name__ == "__main__":
    # eda_dataset_bert()
    main()
