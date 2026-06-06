import math
import os
import statistics as st
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO

import regex

from .perf import perf


class BPETokenizer:
    """
    The actual BPETokenizer implementation,
    what we will end up doing is uhm... some kind of
    accept_pretoken method followed by some kind of
    accept merges?
    """

    def __init__(self, vocab_size: int):
        """
        Construuctor for BPETokenizer,
        vocab_size is the # of tokens you want at the end
        numthreads is the for parallelism

        """
        self.vocab_size = vocab_size
        self.merges: list[tuple[bytes, bytes]] = []

        # Initialize vocab to be {i, byte(i)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # Initialize the pretoken counters
        self.bytecounter: Counter[bytes] = Counter()

    def use_pretoken_counts(self, counter: Counter[str]):
        """
        Will finalize the pretoken counts and get you the final byte counts
        """
        for pretoken in counter:
            bs = pretoken.encode("utf-8")
            for b in bs:
                self.bytecounter[b] += counter[pretoken]

    def _merge(self, byte1: bytes, byte2: bytes):
        pass


def find_chunk_boundaries(
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


RECORD_PERF = True
PRETOKEN_PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| """
    r""""?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
SPECIAL_TOKEN = "<|endoftext|>"


@perf(walltime=True, memory=True, enabled=RECORD_PERF)
def count_pretokens_for_chunk(file: str, start: int, end: int):
    """
    Helper method which will count the pretokens
    for a given chunk

    Arg:
        start: the start of a chunk
        end: the end of a chunk
    """
    counts = Counter()
    with open(file, "rb") as f:
        f.seek(start)

        # Read in single chunks?
        chunk_b = f.read(end - start)
        chunks = chunk_b.decode(encoding="utf-8").split(SPECIAL_TOKEN)
        for chunk in chunks:
            for pretoken in regex.finditer(PRETOKEN_PAT, chunk):
                counts[pretoken.group()] += 1
    return counts


def train(
    tokenizer: BPETokenizer,
    file: str,
    numprocs: int = 4,
    num_chunks: int = 4,
):
    """
    Training loop to train tokenizer, contains the entire split + tokenize +
    ... flow
    """
    with open(file, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_chunks, SPECIAL_TOKEN.encode(encoding="utf-8")
        )

    # Start handing of each chunk to a thread
    with ProcessPoolExecutor(max_workers=numprocs) as pool:
        futures = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            futures.append(pool.submit(count_pretokens_for_chunk, file, start, end))

    total_count = Counter()
    if RECORD_PERF:
        metric_agg = defaultdict(list)
    for future in futures:
        if RECORD_PERF:
            perf_metric, chunk_counter = future.result()
        else:
            chunk_counter = future.result()

        for token in chunk_counter:
            total_count[token] += chunk_counter[token]

        if RECORD_PERF:
            for metric, value in perf_metric.items():
                metric_agg[metric].append(value)

    if RECORD_PERF:
        final_perf = {
            k: {
                "mean": st.mean(vals),
                "median": st.median(vals),
                "stdev": st.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
            for k, vals in metric_agg.items()
        }

        for metric, stats in final_perf.items():
            print(f"{metric}: ")
            for stat_name, stat_val in stats.items():
                print(f"    {stat_name}: {stat_val:.4f}")

    # tokenizer.use_pretoken_counts(total_count)
