# Functions that training and interference share, outside of text manipulation.

import json

seed = 0x62044


def fight_entropy():
    import torch

    torch.manual_seed(seed)

    import random

    random.seed(seed)

    import numpy as np

    np.random.seed(seed)


def fix_mps():
    import torch

    try:
        if not torch.backends.mps.is_available():
            return
    except NameError:
        # No torch, no problem.
        return
    mps_device = torch.device("mps")
    torch.set_default_device(mps_device)
    # This can crash on some versions due to device errors, fast fail so we don't waste time.
    torch.randperm(10, generator=torch.Generator(device=mps_device))


def read_jsonl(fname):
    return [json.loads(l) for l in open(fname)]


def write_jsonl(fname, recs):
    with open(fname, "w") as f:
        for r in recs:
            f.write(json.dumps(r))
            f.write("\n")


def append_jsonl(fname, recs):
    with open(fname, "a") as f:
        for r in recs:
            f.write(json.dumps(r))
            f.write("\n")
