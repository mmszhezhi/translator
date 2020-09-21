import csv
import string
import re
from typing import List, Tuple
from pickle import dump
from unicodedata import normalize
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
import itertools
from pickle import load
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from pickle import load
import random
import tensorflow as tf
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa

# Start of sentence
SOS = "<start>"
# End of sentence
EOS = "<end>"
# Relevant punctuation


PUNCTUATION = set("?,!.")

def load_dataset(filename: str) -> str:
    """
    load dataset into memory
    """
    with open(filename, mode="rt", encoding="utf-8") as fp:
        return fp.read()


def to_pairs(dataset: str, limit: int = None, shuffle=False) -> List[Tuple[str, str]]:
    """
    Split dataset into pairs of sentences, discards dataset line info.

    e.g.
    input -> 'Go.\tGeh.\tCC-BY 2.0 (France) Attribution: tatoeba.org
    #2877272 (CM) & #8597805 (Roujin)'
    output -> [('Go.', 'Geh.')]

    :param dataset: dataset containing examples of translations between
    two languages
    the examples are delimited by `\n` and the contents of the lines are
    delimited by `\t`
    :param limit: number that limit dataset size (optional)
    :param shuffle: default is True
    :return: list of pairs
    """
    assert isinstance(limit, (int, type(None))), TypeError(
        "the limit value must be an integer"
    )
    lines = dataset.strip().split("\n")
    # Radom dataset
    if shuffle is True:
        random.shuffle(lines)
    number_examples = limit or len(lines)  # if None get all
    pairs = []
    for line in lines[: abs(number_examples)]:
        # take only source and target
        src, trg, _ = line.split("\t")
        pairs.append((src, trg))

    # dataset size check
    assert len(pairs) == number_examples
    return pairs


def separe_punctuation(token: str) -> str:
    """
    Separe punctuation if exists
    """

    if not set(token).intersection(PUNCTUATION):
        return token
    for p in PUNCTUATION:
        token = f" {p} ".join(token.split(p))
    return " ".join(token.split())


def preprocess(sentence: str, add_start_end: bool=True) -> str:
    """

    - convert lowercase
    - remove numbers
    - remove special characters
    - separe punctuation
    - add start-of-sentence <start> and end-of-sentence <end>

    :param add_start_end: add SOS (start-of-sentence) and EOS (end-of-sentence)
    """
    re_print = re.compile(f"[^{re.escape(string.printable)}]")
    # convert lowercase and normalizing unicode characters
    sentence = (
        normalize("NFD", sentence.lower()).encode("ascii", "ignore").decode("UTF-8")
    )
    cleaned_tokens = []
    # tokenize sentence on white space
    for token in sentence.split():
        # removing non-printable chars form each token
        token = re_print.sub("", token).strip()
        # ignore tokens with numbers
        if re.findall("[0-9]", token):
            continue
        # add space between words and punctuation eg: "ok?go!" => "ok ? go !"
        token = separe_punctuation(token)
        cleaned_tokens.append(token)

    # rebuild sentence with space between tokens
    sentence = " ".join(cleaned_tokens)

    # adding a start and an end token to the sentence
    if add_start_end is True:
        sentence = f"{SOS} {sentence} {EOS}"
    return sentence


def dataset_preprocess(dataset: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """
    Returns processed database

    :param dataset: list of sentence pairs
    :return: list of paralel data e.g.
    (['first source sentence', 'second', ...], ['first target sentence', 'second', ...])
    """
    source_cleaned = []
    target_cleaned = []
    for source, target in dataset:
        source_cleaned.append(preprocess(source))
        target_cleaned.append(preprocess(target))
    return source_cleaned, target_cleaned


