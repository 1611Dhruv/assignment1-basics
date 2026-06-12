from collections import Counter, defaultdict

from tqdm import tqdm


def train_tokenizer(
    vocab_size: int, special_tokens: list[str] | None, pretoken_counts: Counter
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train tokenizer function which returns the vocabulary and the list of merges (in the order they appear)
    """

    special_token_len = len(special_tokens) if special_tokens is not None else 0

    assert vocab_size >= 256 + special_token_len, f"{vocab_size} is too small"

    token_id = 256
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    to_token_id: dict[bytes, int] = {bytes([i]): i for i in range(256)}
    if special_tokens:
        for special_token in special_tokens:
            special_token_b = special_token.encode(encoding="utf-8")
            vocab[token_id] = special_token_b
            to_token_id[special_token_b] = token_id
            token_id += 1

    bp_heap: BytePairIndexHeap = BytePairIndexHeap()
    merges: list[tuple[bytes, bytes]] = []
    pair_to_set: defaultdict[tuple[bytes, bytes], dict[DLL, bool]] = defaultdict(dict)
    pbar = tqdm(total=vocab_size - token_id, desc="BPE Merges")

    def _add(l_node: DLL):
        if l_node is None or l_node.next is None:
            return

        pair = (l_node.val, l_node.next.val)
        pair_to_set[pair][l_node] = True
        bp_heap.cupsert(l_node.count, pair)

    def _remove(l_node: DLL):
        if l_node is None or l_node.next is None:
            return

        pair = (l_node.val, l_node.next.val)
        # Make sure that this lnode is in the pair
        if l_node in pair_to_set[pair]:
            del pair_to_set[pair][l_node]
            # Update that count
            bp_heap.cupsert(-l_node.count, pair)
            if not pair_to_set[pair]:
                del pair_to_set[pair]

    def build_list(pretoken_b: bytes):
        pretoken = pretoken_b.decode(encoding="utf-8")
        count = pretoken_counts[pretoken]
        head = DLL(bytes([pretoken_b[0]]), count)
        curr = head
        for c in pretoken_b[1:]:
            new_node = DLL(bytes([c]), count)
            new_node.prev = curr
            curr.next = new_node

            _add(curr)
            curr = new_node
        return head

    pretoken_dll = [
        build_list(pretoken.encode("utf-8")) for pretoken in pretoken_counts
    ]
    while bp_heap and token_id < vocab_size:
        _, pair = bp_heap.top()

        # (b1, b2) <- points to node associated with b1
        dlls = pair_to_set[pair].keys()
        new_key = pair[0] + pair[1]
        merges.append(pair)
        vocab[token_id] = new_key
        to_token_id[new_key] = token_id
        token_id += 1
        pbar.update(1)

        for dll in list(dlls):
            if dll.prev is None and dll.next is None:
                continue
            new_dll = DLL(new_key, dll.count)

            _remove(dll)
            if dll.prev is not None:
                _remove(dll.prev)
                dll.prev.next = new_dll
                new_dll.prev = dll.prev
                _add(dll.prev)

            if dll.next.next is not None:
                _remove(dll.next)
                dll.next.next.prev = new_dll
                new_dll.next = dll.next.next
                _add(new_dll)

            # Clean up the dlls
            d1 = dll
            d2 = dll.next

            d1.prev = None
            d1.next = None
            d2.prev = None
            d2.next = None

    pbar.close()
    return vocab, merges


"""
Helper classes to make training faster
"""


class DLL:
    """
    Simple Doubly linked list node for tokenizer training O(1) merges
    """

    def __init__(self, val, count):
        self.val = val
        self.count = count
        self.prev = None
        self.next = None

    def __str__(self):
        res = ""
        if self.prev is not None:
            res += "<->"
        res = f"{res}{self.val}"
        if self.next is not None:
            res += str(self.next)
        return res


class BytePairIndexHeap:
    """
    This heap is used for storing the frequency of a specific byte pair. It supports
    updates to frequency rather than a traditional lazy heap
    """

    def __init__(self):
        self.heap: list[tuple[int, tuple[bytes, bytes]]] = (
            []
        )  # The heap storing all the values
        self.index_map = {}  # Byte pair to the index where the data lives

    def _swap(self, i, j):
        assert i >= 0 and i < len(self.heap), f"Index i({i}) out of bounds"
        assert j >= 0 and j < len(self.heap), f"Index j({j}) out of bounds"

        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.index_map[self.heap[i][1]] = i
        self.index_map[self.heap[j][1]] = j

    def _item_less(self, item1, item2):
        if item1[0] != item2[0]:
            return item1[0] < item2[0]
        return item1[1] < item2[1]

    def _percolate_up(self, i):
        assert i >= 0 and i < len(self.heap), f"Index i({i}) out of bounds"
        while i > 0:
            parent = (i - 1) // 2
            if self._item_less(self.heap[parent], self.heap[i]):
                self._swap(parent, i)
            else:
                break
            i = parent

    def _percolate_down(self, i):
        assert i >= 0 and i < len(self.heap), f"Index i({i}) out of bounds"
        N = len(self.heap)
        while i < N:
            largest = i
            child1 = 2 * i + 1
            child2 = 2 * i + 2
            if child1 < N and self._item_less(self.heap[largest], self.heap[child1]):
                largest = child1

            if child2 < N and self._item_less(self.heap[largest], self.heap[child2]):
                largest = child2

            if largest == i:
                break

            self._swap(i, largest)
            i = largest

    def cupsert(self, del_count: int, pair: tuple[bytes, bytes]):
        """
        Insert of Updates the item to the heap. Note that this will just replace
        the binary heap count
        """
        if pair in self.index_map:
            i = self.index_map[pair]
            old_item = self.heap[i]
            count = old_item[0] + del_count
            if count < 0:
                print(del_count, pair)
                print(self.heap[:10])
            assert count >= 0, "Count of upserted items must be >= 0"
            item = (count, pair)

            self.heap[i] = item
            if del_count > 0:
                self._percolate_up(i)
            else:
                # It was a removal of count

                # if the new count is 0, it must be removed from the heap
                if count == 0:
                    # If not the last one,
                    if i != len(self.heap) - 1:
                        self._swap(i, len(self.heap) - 1)
                        self._percolate_down(i)
                    self.heap.pop()
                    del self.index_map[pair]
                else:
                    self._percolate_down(i)
        else:
            count = del_count
            if count < 0:
                print(del_count, pair)
                print(self)
            assert count >= 0, "Count of upserted items must be >= 0"
            item = (count, pair)

            if count == 0:
                return
            i = len(self.heap)
            self.heap.append(item)
            self.index_map[pair] = i
            self._percolate_up(i)

    def top(self) -> tuple[int, tuple[bytes, bytes]]:
        assert not self.empty(), "Can't call top on an empty heap"
        return self.heap[0]

    def pop(self) -> tuple[int, tuple[bytes, bytes]]:
        """
        Returns the largest item in the heap
        """
        N = len(self.heap)
        assert N > 0, "Can't call pop on an empty heap"
        if N > 1:
            self._swap(0, N - 1)
            top_item = self.heap.pop()
            del self.index_map[top_item[1]]
            self._percolate_down(0)
        else:
            top_item = self.heap.pop()
            del self.index_map[top_item[1]]

        return top_item

    def empty(self):
        """
        Returns true if heap is empty
        """
        return len(self.heap) == 0

    def __bool__(self):
        return not self.empty()

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return f"Heap: {self.heap}\n Index_map: {self.index_map}"
