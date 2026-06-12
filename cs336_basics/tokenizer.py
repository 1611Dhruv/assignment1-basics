import pickle
from abc import ABC, abstractclassmethod, abstractmethod
from functools import reduce
from typing import Iterable, Iterator

import regex

PRETOKEN_REG = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
PRETOKEN_PAT = regex.compile(PRETOKEN_REG)


class Tokenizer(ABC):
    @abstractclassmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ): ...
    @abstractmethod
    def encode(self, text: str) -> list[int]: ...
    @abstractmethod
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]: ...
    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...


class BPETokenizer(Tokenizer):
    """
    The actual BPETokenizer implementation,
    what we will end up doing is uhm... some kind of
    accept_pretoken method followed by some kind of
    accept merges?
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a BPE tokenizer out of vocabulary merges and optional
        special_tokens list
        """
        self.vocab = vocab
        self.to_token_id = {vocab[token_id]: token_id for token_id in vocab}
        self.to_merge = {pair: order for order, pair in enumerate(merges)}
        self.special_pattern = "(?!)"
        self.MAX_SPECIAL = 1

        if special_tokens:
            self.MAX_SPECIAL = len(max(special_tokens, key=len))
            self.special_pattern = "|".join(
                regex.escape(t)
                for t in sorted(
                    special_tokens,
                    key=len,
                    reverse=True,
                )
            )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)

        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)

        return BPETokenizer(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: str) -> Iterator[int]:
        pretoken_b = pretoken.encode(encoding="utf-8")
        pretoken_bl = [bytes([ch]) for ch in pretoken_b]
        while len(pretoken_bl) > 1:
            best = min(
                (
                    (self.to_merge.get(pair, float("inf")), i)
                    for i, pair in enumerate(zip(pretoken_bl, pretoken_bl[1:]))
                ),
            )
            rank, i = best
            if rank == float("inf"):
                break
            pretoken_bl = (
                pretoken_bl[:i]
                + [pretoken_bl[i] + pretoken_bl[i + 1]]
                + pretoken_bl[i + 2 :]
            )

        for b in pretoken_bl:
            yield self.to_token_id[b]

    def encode(self, text: str) -> list[int]:
        """
        Encodes a piece of text / chunk into a list of tokens
        """
        chunk_size = 4096 // 4
        chunks = (text[i : i + chunk_size] for i in range(0, len(text), chunk_size))
        return list(self.encode_iterable(chunks))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encodes in an iterable fashion
        """
        buf = ""
        special_pat = regex.compile(self.special_pattern)
        for chunk in iterable:
            buf += chunk

            last = 0
            for m in special_pat.finditer(buf):
                if m.end() == len(buf):
                    break
                yield from self._pretokenize(buf[last : m.start()], final=True)
                yield self.to_token_id[m.group().encode("utf-8")]
                last = m.end()
            buf = buf[last:]

            safe_end = len(buf) - (self.MAX_SPECIAL - 1)
            if safe_end > 0:
                safe, buf = buf[:safe_end], buf[safe_end:]
                tail = yield from self._pretokenize(safe, final=False)
                buf = tail + buf
        last = 0
        for m in special_pat.finditer(buf):
            yield from self._pretokenize(buf[last : m.start()], final=True)
            yield self.to_token_id[m.group().encode("utf-8")]
            last = m.end()
        yield from self._pretokenize(buf[last:], final=True)

    def _pretokenize(self, text, final):
        last = 0
        for m in PRETOKEN_PAT.finditer(text):
            if not final and m.end() == len(text):
                return text[m.start() :]
            yield from self._encode_pretoken(m.group())
            last = m.end()
        if final:
            return ""
        return text[last:]

    def decode(self, ids: list[int]) -> str:
        """
        Decodes form a list of tokens using the vocabulary
        """
        return b"".join([self.vocab[i] for i in ids]).decode(
            encoding="utf-8",
            errors="replace",
        )
