import functools
import time

from cs336_basics.tokenizer import BPETokenizer, train

"""
For perf testing
"""

TEST_FILE = "data/my_data.txt"
BIG_FILE = "data/owt_train.txt"

FILE = BIG_FILE

tokenizer = BPETokenizer(100)


def labeled(label, nosleep=False):
    if not nosleep:
        time.sleep(100)
    print(f"{'='* 30}")
    print(f"Calling {label}:")


## 10GB memory and process ablation
labeled("24 proc, 10GB", nosleep=True)
train(tokenizer, FILE, numprocs=24, peak_mem=10 * 2**30)
labeled("12 proc, 10GB")
train(tokenizer, FILE, numprocs=12, peak_mem=10 * 2**30)
labeled("8 proc, 10GB")
train(tokenizer, FILE, numprocs=8, peak_mem=10 * 2**30)
labeled("4 proc, 10GB")
train(tokenizer, FILE, numprocs=4, peak_mem=10 * 2**30)

## 24 procs and my abla
labeled("24 proc, 10GB")
train(tokenizer, FILE, numprocs=24, peak_mem=10 * 2**30)
labeled("24 proc, 5GB")
train(tokenizer, FILE, numprocs=24, peak_mem=5 * 2**30)
labeled("24 proc, 2GB")
train(tokenizer, FILE, numprocs=24, peak_mem=2 * 2**30)
labeled("24 proc, 1GB")
train(tokenizer, FILE, numprocs=24, peak_mem=2**30)
