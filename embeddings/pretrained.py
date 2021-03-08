import os
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder


def load_elmo():
    os.system("wget http://vectors.nlpl.eu/repository/11/195.zip")
    os.system("mkdir elmo && mv 195.zip elmo/195.zip && cd elmo && unzip 195.zip && rm 195.zip && cd ..")
    # if use for training model
    # elmo = Elmo(options_file="elmo/options.json",
    # weight_file="elmo/model.hdf5", num_output_representations=1, dropout=0)
    # if use only embeddings
    elmo = ElmoEmbedder(options_file="elmo/options.json", weight_file="elmo/model.hdf5", cuda_device=0)
    return elmo


def get_elmo_text_embedding(text, elmo):
    """ Returns embedding for text (by averaging) where text = List[sent], sent=List[token_text]"""
    embeddings = elmo.batch_to_embeddings(text)[0].cpu().numpy()  # (len(texts), 3, len(text), 1024)
    embeddings = embeddings.swapaxes(1, 2)  # (len(texts), len(text), 3, 1024)
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1)  # (len(texts), len(text), 3072)
    embeddings = np.mean(embeddings, axis=1)  # (len(texts), 3072) - average by sentence
    embeddings = np.mean(embeddings, axis=0)  # (1, 3072) - average by text
    return embeddings

# TODO: add BERT embeddings
