from collections import Counter
import sys


import ctranslate2

import transformers

from mavis.stc import spell_checker
from mavis.corrector import common, text


class InferencerCt2(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.translator = ctranslate2.Translator(
            self.model_path, device="cpu", inter_threads=2, intra_threads=2
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            from_slow=True,
            legacy=True,
            local_files_only=True,
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
