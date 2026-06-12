import codecs
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, BinaryIO, Callable, Iterator

import numpy as np
import regex
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer

"""
Helpers to pretokenize
"""

PRETOKEN_REG = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
PRETOKEN_PAT = regex.compile(PRETOKEN_REG)
SPECIAL_EOT_TOKEN = "<|endoftext|>"


def multiproc_pretokens(
    file: str,
    mapper: Callable[[str | os.PathLike, int, int], Any],
    reducer: Callable[[Any, Any], Any],
    reducer_init: Any,
    end_token: str = SPECIAL_EOT_TOKEN,
    numprocs: int = 4,
):
    """
    Training loop to train tokenizer, contains the entire split + tokenize +
    ... flow
    """
    num_chunks: int = 4 * numprocs

    with open(file, "rb") as f:
        boundaries = _find_chunk_boundaries(
            f, num_chunks, end_token.encode(encoding="utf-8")
        )

    # Start handing of each chunk to a thread
    with ProcessPoolExecutor(max_workers=numprocs) as pool:
        futures = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            futures.append(pool.submit(mapper, file, start, end))

        reducer_result = reducer_init
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="pretokenizing in parallel"
        ):
            future_result = future.result()
            reducer_result = reducer(reducer_result, future_result)

    return reducer_result


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_pretokens_mapper(file: str, start: int, end: int, end_token: str):
    """
    Helper method which will count the pretokens
    for a given chunk in a streaming fashion

    Arg:
        start: the start of a chunk
        end: the end of a chunk
    """

    counts = Counter()
    PAGE_SZ = 4096
    end_token_b = end_token.encode(encoding="utf-8")

    remaining = end - start
    buf = b""
    with open(file, "rb") as f:
        f.seek(start)
        while remaining > 0:
            page = f.read(min(remaining, PAGE_SZ))
            if not page:
                break

            remaining -= len(page)
            buf += page

            while (idx := buf.find(end_token_b)) != -1:
                doc = buf[:idx].decode(encoding="utf-8")
                for m in regex.finditer(PRETOKEN_PAT, doc):
                    counts[m.group()] += 1
                buf = buf[idx + len(end_token_b) :]
    if buf:
        doc = buf.decode(encoding="utf-8")
        for m in regex.finditer(PRETOKEN_PAT, doc):
            counts[m.group()] += 1

    return counts


def count_pretokens_reducer(state: Counter, partial: Counter) -> Counter:
    """
    Pretokenize reducer to reduce all the parts
    """
    state.update(partial)
    return state


def read_chunk(path: str, start: int, end: int, size: int = 4096) -> Iterator[str]:
    """
    Reads the chunk based on size
    """
    with open(path, "rb") as f:
        decoder = codecs.getincrementaldecoder("utf-8")()
        remaining = end - start
        while remaining > 0:
            page = f.read(min(size, remaining))
            if not page:
                break
            remaining -= len(page)
            text = decoder.decode(page)
            if text:
                yield text
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail


def encode_pretokenize_mapper(
    file: str, start: int, end: int, tokenizer: Tokenizer
) -> tuple[int, np.ndarray]:
    """
    Encoding parital mapper
    """
    return start, np.fromiter(
        tokenizer.encode_iterable(read_chunk(file, start, end)),
        dtype=np.uint16,
    )


def encode_pretokenize_reducer(
    state: list[tuple[int, np.ndarray]], partial: tuple[int, np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    """
    Encoding parital reducer
    """
    state.append(partial)
    return state
