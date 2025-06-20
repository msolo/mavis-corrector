#!/usr/bin/env python3

import argparse

import transformers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    # I vastly prefer - to _, but just be consistent with the other ct2 tools.
    ap.add_argument("--output_dir")

    args = ap.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        from_slow=True,
        legacy=True,
        local_files_only=False,
    )
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
