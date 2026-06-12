import pickle
from collections import Counter
from functools import partial
from random import randint, shuffle
from timeit import timeit

import numpy as np
import regex as re

from cs336_basics.pretokenize import (
    SPECIAL_EOT_TOKEN,
    count_pretokens_mapper,
    count_pretokens_reducer,
    encode_pretokenize_mapper,
    encode_pretokenize_reducer,
    multiproc_pretokens,
)
from cs336_basics.tokenizer import BPETokenizer, Tokenizer
from cs336_basics.train.tokenizer import train_tokenizer

TINY_STORIES_FILE = "data/TinyStoriesV2-GPT4-train.txt"
OPEN_WEB_TEXT_FILE = "data/owt_train.txt"
SMALL_DATA = "data/my_data.txt"
EDGE = "data/my_data1.txt"

FILE = OPEN_WEB_TEXT_FILE
NPROC = 24


def train_and_save_pretok(
    file: str,
    nproc: int,
    vocab_size: int,
    special_tokens: list[str] | None,
    label: str = "cool",
):
    pretoken_c = multiproc_pretokens(
        file,
        partial(count_pretokens_mapper, end_token=SPECIAL_EOT_TOKEN),
        count_pretokens_reducer,
        Counter(),
        numprocs=nproc,
    )
    vocab, merges = train_tokenizer(
        vocab_size,
        special_tokens,
        pretoken_c,
    )

    with open(f"models/vocab_{label}.pkl", "wb") as vocab_file:
        pickle.dump(vocab, vocab_file)

    with open(f"models/merges_{label}.pkl", "wb") as merge_file:
        pickle.dump(merges, merge_file)


def encode_data(file: str, nproc: int, tokenizer: Tokenizer, label: str = "cool"):
    chunk_encoded = multiproc_pretokens(
        file,
        partial(encode_pretokenize_mapper, tokenizer=tokenizer),
        encode_pretokenize_reducer,
        [],
        numprocs=nproc,
    )
    chunk_encoded.sort(key=lambda t: t[0])
    with open(f"models/encoded_data_{label}", "wb") as encode_data_file:
        for _, chunk in chunk_encoded:
            chunk.tofile(encode_data_file)


owl_tok = BPETokenizer.from_files(
    "models/vocab_owl.pkl",
    "models/merges_owl.pkl",
    special_tokens=[SPECIAL_EOT_TOKEN],
)

tiny_tok = BPETokenizer.from_files(
    "models/vocab_tiny.pkl",
    "models/merges_tiny.pkl",
    special_tokens=[SPECIAL_EOT_TOKEN],
)

data = np.fromfile("models/encoded_data_tiny_tokenizer", dtype=np.uint16)
print(data)
print(tiny_tok.decode(data[:100]))
encode_data(OPEN_WEB_TEXT_FILE, NPROC, tiny_tok, "tiny_tokenizer")
