from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import functools
import itertools
import json
import math
import os
import re
import subprocess
import sys
import types


import ctranslate2
import numpy as np
import torch

import transformers
from transformers import (
    GenerationConfig,
)

from mavis.stc import spell_checker
from mavis.corrector import common, text


@dataclass
class InferencerHF:
    model_path: str

    use_half: bool = False
    use_onnx: bool = False
    use_xla: bool = False

    tokenizer = None
    model = None

    # You won't get far without a tokenizer and these seem to be shared amongst fork() calls.
    def __post_init__(self):
        # Without MPS, this will be slow.
        common.fix_mps()
        input_truncation_side = "right"

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            legacy=False,
            truncation_side=input_truncation_side,
            from_slow=True,
            local_files_only=True,
        )

        model_args = {}

        if self.use_half:
            model_args["torch_dtype"] = torch.bfloat16

        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            **model_args,
        )

        # for some reason, this ends up on the wrong place
        self.model.to(device=torch.get_default_device())

    def make_generation_config(self, strategy, **generation_args):
        gargs = {
            # # Words and tokens are not 1:1, and we probably can't predict more than 25 words in a sentence,
            # # so give us some headroom.
            # 'max_new_tokens': int(min_new_tokens * 1.5),
            # 'min_new_tokens': min_new_tokens,
            "num_return_sequences": 128,
            "temperature": 1.0,
            "stop_strings": ["</s>"],
        }
        if strategy == "topk":
            # Follow SpeakFaster
            gargs["top_k"] = 40
            gargs["do_sample"] = True
            # early_stopping should only apply to beam search anyway.
            gargs["early_stopping"] = False
        elif strategy == "beam":
            gargs["num_beams"] = gargs["num_return_sequences"]
            gargs["early_stopping"] = True
        else:
            raise Exception("invalid strategy", strategy)

        gargs.update(generation_args)
        generation_config = GenerationConfig.from_pretrained(
            self.model_path, local_files_only=True, **gargs
        )
        return generation_config

    # FIXME: maybe memoize
    def generate(self, texts, strategy="topk", **generation_args):
        # Prepare the input text with the "grammar: " prefix
        if isinstance(texts, str):
            texts = [texts]
        input_texts = ["grammar: " + t for t in texts]

        input_ids = self.tokenizer(
            input_texts,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )["input_ids"]

        # try to add 20% more tokens, just in case we split things.
        input_token_count = input_ids.shape[-1]
        extra_tokens = max(input_token_count * 0.2, 1)
        max_new_tokens = max(32, input_token_count + extra_tokens)
        generation_config = self.make_generation_config(
            strategy,
            min_new_tokens=1,
            max_new_tokens=max_new_tokens,
            **generation_args,
        )
        outputs = self.model.generate(
            input_ids, tokenizer=self.tokenizer, generation_config=generation_config
        )

        for tl in outputs:
            if self.tokenizer.eos_token_id in tl:
                res = self.tokenizer.decode(tl).replace("<pad>", "").strip()
                if not res.endswith("</s>"):
                    print("found odd EOS", res)

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_outputs

    def correct(self, text, strategy="topk", **generation_args):
        input_texts = [text, text.replace(" ", "")]
        results = self.generate(input_texts, strategy=strategy, **generation_args)
        results = top5(results, text)
        return fix_terminal_punctuation(results, text)


class InferencerCt2(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.translator = ctranslate2.Translator(
            self.model_path, device="cpu", inter_threads=2, intra_threads=2
        )
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                "willwade/t5-small-spoken-typo",
                from_slow=True,
                legacy=True,
                local_files_only=True,
            )
        except EnvironmentError:
            print("falling back to downloading tokenizer", file=sys.stderr)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                "willwade/t5-small-spoken-typo",
                from_slow=True,
                legacy=True,
                local_files_only=False,
            )

    def detokenize(self, tl):
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tl))

    def make_generation_config(self, strategy, **generation_args):
        gargs = {
            # # Words and tokens are not 1:1, and we probably can't predict more than 25 words in a sentence,
            # # so give us some headroom.
            # 'max_new_tokens': int(min_new_tokens * 1.5),
            # 'min_new_tokens': min_new_tokens,
            "num_return_sequences": 128,
            # temp 0.9 seems to work better.
            "temperature": 0.9,
            "top_k": 40,
            "stop_strings": ["</s>"],
        }
        gargs.update(generation_args)
        return gargs

    def generate(self, texts, strategy="topk", **generation_args):
        # Prepare the input text with the "grammar: " prefix
        if isinstance(texts, str):
            texts = [texts]
        input_texts = ["grammar: " + t for t in texts]
        # This has an odd API of tokenized strings.
        input_tokens = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(t))
            for t in input_texts
        ]
        gargs = self.make_generation_config(strategy, **generation_args)
        results = self.translator.translate_batch(
            input_tokens,
            beam_size=1,
            sampling_topk=gargs["top_k"],
            num_hypotheses=gargs["num_return_sequences"],
            sampling_temperature=gargs["temperature"],
            max_batch_size=1,  # setting match_batch_size to 1 seems to improve performance.
        )
        all_results = [
            self.detokenize(tl) for tl in sum([r.hypotheses for r in results], [])
        ]
        return all_results

    def correct(self, text, strategy="topk", **generation_args):
        input_texts = [text, text.replace(" ", "")]
        results = self.generate(input_texts, strategy=strategy, **generation_args)
        results = top5(results, text)
        return fix_terminal_punctuation(results, text)


def fix_terminal_punctuation(results, original_text):
    if original_text.strip().endswith(tuple(".?!")):
        punc = original_text.strip()[-1]
        results = [t + punc for t in results]
    return results


def filter_candidates(candidates, original):
    original_errors = spell_checker.misspellings(original)
    allow_question = "?" in original
    allow_exclamation = "!" in original
    filtered_candidates = []
    # FIXME: we could filter candidates with garbage words unless the appear in the original
    # This filtering is too aggressive.
    for i, x in enumerate(candidates):
        # if "?" in x:
        #     if allow_question:
        #         filtered_candidates.append(x)
        # elif "!" in x:
        #     if allow_exclamation:
        #         filtered_candidates.append(x)
        if not x.strip():  # apparently we get empty generation?
            print("empty generation for", i, original, candidates)
            continue
        # No sense in letting our temperature interfere, filter things that add more spelling errors.
        if len(spell_checker.misspellings(x)) > len(original_errors):
            continue
        filtered_candidates.append(x)
    return filtered_candidates


def top5(candidates, original):
    """Return the top-5 normalized candidates."""
    filtered_candidates = filter_candidates(candidates, original)
    # Counter uses the original order of items in the case of a tie.
    c = Counter((text.norm_text_eval(x) for x in filtered_candidates))
    results = [x[0] for x in c.most_common(5)]
    return results
