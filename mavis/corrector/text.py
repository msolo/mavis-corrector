# Handle text normalization and cleanup

import re

text_emoji = [":)", ":(", ";)", ":/", ":-/", ":-)", ";-)", ":-(", ":'(", ":p", ":P"]


def _normalize_text(t):
    # Stupefy smart quotes
    t = t.replace("“", "")  # smart quote
    t = t.replace("”", "")
    t = t.replace("‘", "'")  # smart apostrophe
    t = t.replace("’", "'")
    t = t.replace("\u2013", "-")  # one of the dashes -
    t = t.replace("\u2014", "-")  # one of the dashes -
    # remove quotes? these aren't useful for our case per-se, but maybe they clean up noise from some data sets?
    t = t.replace('"', "")
    # remove newlines, i think it can confuse the q&a formatting.
    t = t.replace("\n", " ")
    t = t.replace("\t", " ")
    t = " ".join([w for w in t.split() if w not in text_emoji])
    return t


# prune out known speaker / texting issues.
double_dot = re.compile(r"(\w)\.\. ")
many_dot = re.compile(r"\.{3,}")
# common text typo
trailing_char = re.compile(" [nb]$")
comma_norm = re.compile(
    r"\b,\b",
)
hyphen_norm = re.compile(
    r"\b\s*-+\s*\b",
)
equals_norm = re.compile(
    r"\b\s*=+\s*\b",
)

whitespace = re.compile(r"\s{2,}")
trailing_punctuation = re.compile(r"\s+([?!.])$")
parenthetical_noise = re.compile(r"[(][^)]*[)]")
bracket_noise = re.compile(r"[\[][^\]]*[\]]")
ddc_punctuation = re.compile(r"(\w)\s([.,!?'])")
ddc_currency = re.compile(r"([$])\s(\d)")
stray_html = re.compile(r"<[/]?[uib]>", re.IGNORECASE)


def norm_text(t):
    t = _normalize_text(t)
    t = comma_norm.sub(", ", t)
    t = hyphen_norm.sub(" - ", t)
    t = equals_norm.sub(" ", t)  # no real reason for this
    # double punctuation usually an error
    t = t.replace("''", "'")  # apostrophe
    # t = t.replace("--", "-") # hyphen dash

    t = ddc_punctuation.sub(r"\1\2", t)
    t = ddc_currency.sub(r"\1\2", t)
    t = parenthetical_noise.sub("", t)
    t = bracket_noise.sub("", t)
    t = stray_html.sub("", t)
    # Normalize double dot to a single period.
    t = double_dot.sub(r"\1. ", t)
    # Normalize many dots to elipsis and make sure there is a space to kind of split things up.
    t = many_dot.sub(" ... ", t)
    t = trailing_char.sub("", t)
    t = trailing_punctuation.sub(r"\1", t)
    t = whitespace.sub(" ", t)

    # DailyDialog has notoriously odd spacing.
    # FIXME: SentenceSplitter is pretty bad and this normalization doens't seem to be helping.
    # It's not worth the NLTK headache.
    # t = ' '.join(SentenceSplitter.split(t))
    t = t.strip()
    return t


# speech doesn't care in some cases, maybe we shouldn't either...
homophone_map = {
    "it's": "its",
}


def norm_text_eval(t):
    # remove trailing punctuation
    t = norm_text(t).strip(".?!").lower()
    # speech synth sometimes varies "dont" vs "don't" but these sound identical and are common errors
    homophone_map = {
        "it's": "its",
    }
    return " ".join([homophone_map.get(w, w) for w in t.split()])


_punct = r"""/\|,./<>?;:!@#$%^&*()-_=+'"¡¿…`‘“”’[]{}"""


# Filter things that are just punctuation
def is_punctuation(c):
    return c in _punct


def is_numeric_or_punctuation(text):
    for c in text:
        if not c.isnumeric() and not is_punctuation(c):
            return False
    return True


def is_numeric_with_punctuation(text):
    has_num = False
    has_punc = False
    for c in text:
        if c.isnumeric():
            has_num = True
        elif is_punctuation(c):
            has_punc = True
        else:
            return False
    return has_num


def fmt_input(sentence, context=()):
    if context:
        ctx_str = " ".join(context)
        return f"Correct sentence: {sentence} Context: {ctx_str}"
    return f"Correct sentence: {sentence}"
