import re
import json
import math
import itertools
import pkgutil
from collections import Counter, OrderedDict
from multiprocessing import Process, Queue

import wordsegment as ws


ws.load()
UNIGRAMS = Counter(ws.UNIGRAMS)
BIGRAMS = Counter(ws.BIGRAMS)
# Magic total is "1024908267229.0". This comes from here: https://catalog.ldc.upenn.edu/LDC2006T13a
# wordsegment uses that, since the unigrams and bigrams are top-n samples of those datasets.
TOTAL = ws.Segmenter.TOTAL


def json_load(filename):
    return json.loads(pkgutil.get_data(__name__, filename))


SPELL_UNIGRAMS = Counter(json_load("en.json"))
SPELL_TOTAL = SPELL_UNIGRAMS.total()

WORDS = ALL_WORDS = SPELL_UNIGRAMS
PROPER_NAMES = {}

# # Try using separate sets for regular and proper nouns.
# # This didn't yield immediate improvements though.
# _ALL_WORDS = {x.strip() for x in pkgutil.get_data(__name__, filename).splitlines()}
# PROPER_NAMES = {w.lower() for w in _ALL_WORDS if w[0].isupper()}
# WORDS = {w.lower() for w in _ALL_WORDS if w[0].islower()}
# ALL_WORDS = WORDS | PROPER_NAMES

contractions = """I'd,I had / I would
I'll,I will
I'm,I am
I've,I have
aren't,are not
can't,cannot
could've,could have
couldn't,could not
daren't,dare not
didn't,did not
doesn't,does not
don't,do not
hadn't,had not
hasn't,has not
haven't,have not
he'd,he had / he would
he'll,he will
he's,he is / he has
here's,here is
how'll,how will
how's,how is / how has
isn't,is not
it'd,it had / it would
it'll,it will
it's,it is / it has
let's,let us
might've,might have
mightn't,might not
must've,must have
mustn't,must not
needn't,need not
oughtn't,ought not
shan't,shall not
she'd,she had / she would
she'll,she will
she's,she is / she has
should've,should have
shouldn't,should not
that'll,that will
that's,that is / that has
there'd,there would / there had
there'll,there will
there's,there is / there has
there've,there have
they'd,they would / they had
they'll,they will
they're,they are
they've,they have
wasn't,was not
we'd,we would / we had
we're,we are
we've,we have
weren't,were not
what'll,what will
what're,what are
what's,what is / what has
what've,what have
where's,where is / where has
who'd,who would / who had
who'll,who will
who're,who are
who's,who is / who has
who've,who have
won't,will not
would've,would have
wouldn't,would not
you'd,you had / you would
you'll,you will
you're,you are
you've,you have
"""

for line in contractions.lower().splitlines():
    contraction, expansion = line.lower().strip().split(",")
    # Use only the first expansion
    expansion = expansion.split(" / ")[0].strip()
    norm_contraction = contraction.replace("'", " ").strip()
    BIGRAMS[norm_contraction] = BIGRAMS[expansion]


# https://userinterfaces.aalto.fi/136Mkeystrokes/resources/chi-18-analysis.pdf


def update_proper_nouns(names):
    # for names, make sure none of them are too rare.
    # pick something decent and then write fake counts.
    _names = list(sorted(names, key=lambda x: UNIGRAMS[x]))
    name_count = UNIGRAMS[_names[-5]]
    for name in _names:
        UNIGRAMS[name] = name_count

    _names = list(sorted(names, key=lambda x: SPELL_UNIGRAMS[x]))
    name_count = SPELL_UNIGRAMS[_names[-5]]
    for name in _names:
        SPELL_UNIGRAMS[name] = name_count


def product(nums):
    "Multiply the numbers together.  (Like `sum`, but with multiplication.)"
    result = 1
    for x in nums:
        result *= x
    return result


def pdist_additive_smoothed(counter, total, c=1):
    """The probability of word, given evidence from the counter.
    Add c to the count for each item, plus the 'unknown' item."""
    # total = Amount of evidence
    Nplus = total + c * (len(counter) + 1)  # Evidence plus fake observations

    def Pword(word):
        return (counter[word] + c) / Nplus

    return Pword


def P1w(word):
    return UNIGRAMS[word] / TOTAL


P1w_smooth = pdist_additive_smoothed(UNIGRAMS, TOTAL)
P1w_spell_smooth = pdist_additive_smoothed(SPELL_UNIGRAMS, SPELL_TOTAL)


def P2w(bigram):
    return BIGRAMS[bigram] / TOTAL


def score_match(word, edits=0):
    if use_fine_distance:
        lp = math.log10(P1w_smooth(word))
        return (edits * lp, lp, edits)
    else:
        return math.log10(P1w_spell_smooth(word))


use_fine_distance = False


def _correct_word_topn(word, n=5, return_scores=False, score_all_candidates=False):
    "Find the best spelling correction for this word."

    candidates = []
    if word in WORDS:
        candidates.append((word, 0))
    if not candidates or score_all_candidates:
        if use_fine_distance:
            for c, dist in edits1_scored(word):
                if known([c]):
                    candidates.append((c, dist))
        else:
            for c in known(edits1(word)):
                candidates.append((c, 1))
    if word in PROPER_NAMES:
        candidates.append((word, 0.95))
    if not candidates or score_all_candidates:
        if use_fine_distance:
            for c, dist in edits2_scored(word):
                if known([c]):
                    candidates.append((c, dist))
        else:
            for c in known(edits2(word)):
                candidates.append((c, 2))
    if len(candidates) < 2 or score_all_candidates:
        candidates.append((word, 3))

    # make sure smallest edit wins
    candidates = dict(reversed(candidates))

    # FIXME: dirty
    if (
        len(word) > 3
        and "the" in candidates
        and len([c for c, d in candidates.items() if d < 3]) > 1
    ):
        # "the" is so common, this will always win
        del candidates["the"]

    scored = [(word, score_match(word, edits)) for word, edits in candidates.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = itertools.islice(scored, n)
    if return_scores:
        return list(scored)
    else:
        return [x[0] for x in scored]


def known(words):
    "Return the subset of words that are actually in the dictionary."
    # This is completely key to quality for some reason.
    return {w for w in words if w in ALL_WORDS}


def edits2(word):
    "Return all strings that are two edits away from this word."
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


def edits2_scored(word):
    "Return all strings that are two edits away from this word."
    # Nested comprehendsions suck.
    d = {}
    for _word1, _dist1 in edits1_scored(word):
        for word2, _dist2 in edits1_scored(_word1):
            d[word2] = _dist1 + _dist2
    return list(d.items())


alphabet = "abcdefghijklmnopqrstuvwxyz"
qwerty_layout = {
    "a": ["q", "w", "s", "z"],
    "b": ["v", "g", "h", "n"],
    "c": ["x", "d", "f", "v"],
    "d": ["s", "e", "r", "f", "c", "x"],
    "e": ["w", "r", "d", "s"],
    "f": ["d", "r", "t", "g", "v", "c"],
    "g": ["f", "t", "y", "h", "b", "v"],
    "h": ["g", "y", "u", "j", "n", "b"],
    "i": ["u", "o", "k", "j"],
    "j": ["h", "u", "i", "k", "m", "n"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "m": ["n", "j", "k", "l"],
    "n": ["b", "h", "j", "m"],
    "o": ["i", "p", "l", "k"],
    "p": ["o", "l"],
    "q": ["w", "a"],
    "r": ["e", "t", "f", "d"],
    "s": ["a", "w", "e", "d", "x", "z"],
    "t": ["r", "y", "g", "f"],
    "u": ["y", "i", "j", "h"],
    "v": ["c", "f", "g", "b"],
    "w": ["q", "e", "s", "a"],
    "x": ["z", "s", "d", "c"],
    "y": ["t", "u", "h", "g"],
    "z": ["a", "s", "x"],
    "'": [],
}


# Return list of likely characters to substitute based on qwerty layout.
def char_subs(c):
    return qwerty_layout[c]


def edits1(word):
    "Return all strings that are one edit away from this word."
    pairs = splits(word)
    deletes = [a + b[1:] for (a, b) in pairs if b]
    transposes = [a + b[1] + b[0] + b[2:] for (a, b) in pairs if len(b) > 1]
    replaces = [a + c + b[1:] for (a, b) in pairs for c in alphabet if b]
    inserts = [a + c + b for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def edits1_scored(word):
    "Return all strings that are one edit away from this word."
    pairs = splits(word)
    deletes = [a + b[1:] for (a, b) in pairs if b]
    transposes = [a + b[1] + b[0] + b[2:] for (a, b) in pairs if len(b) > 1]
    likely_replaces = []
    for a, b in pairs:
        if b:
            for c in char_subs(b[0]):
                likely_replaces.append(a + c + b[1:])

    replaces = [a + c + b[1:] for (a, b) in pairs for c in alphabet if b]
    inserts = [a + c + b for (a, b) in pairs for c in alphabet]

    double_inserts = [a + b[0] + b for (a, b) in pairs if b]
    apostrophe_inserts = []
    if len(word) > 1:
        apostrophe_inserts.append(word[:-1] + "'" + word[-1:])
    if len(word) > 2:
        apostrophe_inserts.append(word[:-2] + "'" + word[-2:])

    # FIXME scoring all edits the same might be wrong, transposes are probably much more likely
    # It's also a bonus is all the characters are in the word, just jumbled. reic -> rice
    scored = {}
    for x in transposes:
        if x not in scored:
            scored[x] = 0.7
    for x in likely_replaces:
        if x not in scored:
            scored[x] = 0.8
    for x in double_inserts:
        if x not in scored:
            scored[x] = 0.8
    for x in apostrophe_inserts:
        if x not in scored:
            scored[x] = 1.0
    for x in replaces:
        if x not in scored:
            scored[x] = 1.0
    for x in deletes + inserts:
        if x not in scored:
            scored[x] = 1.0
    scored_items = list(scored.items())
    return scored_items


def correct_text(text):
    "Correct all the words within a text, returning the corrected text."
    return word_re.sub(correct_match, text)


def correct_match(match):
    "Spell-correct word in match, and preserve proper upper/lower/title case."
    word = match.group()
    new_word = correct_word(word.lower())
    # print('correct_match', match, 'word:', word, 'lower:', word.lower(), 'new:', new_word)
    return case_of(word)(new_word)


def case_of(text):
    "Return the case-function appropriate for text: upper, lower, title, or just str."
    return (
        str.upper
        if text.isupper()
        else str.lower if text.islower() else str.title if text.istitle() else str
    )


# Segmentation - we implement something similar to wordsegment, but we'll handle contractions.
def splits(text, start=0, L=20):
    "Return a list of all (first, rest) pairs; start <= len(first) <= L."
    return [(text[:i], text[i:]) for i in range(start, min(len(text), L) + 1)]


# Change Pwords to use P1w (the bigger dictionary) instead of Pword
def Pwords(words, pword=P1w):
    "Probability of words, assuming each word is independent of others."
    return product(pword(w) for w in words)


def log_prob_words(words, pword=P1w_smooth):
    "Probability of words, assuming each word is independent of others."
    return sum(math.log10(pword(w)) for w in words)


# Limit growth of memoization
class BoundedLRU(OrderedDict):
    "Store items in the order the keys were last used."

    def __init__(self, bound):
        super().__init__(self)
        self.bound = bound

    def __getitem__(self, key):
        v = super().__getitem__(key)
        self.move_to_end(key)
        return v

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        while len(self) > self.bound:
            self.popitem(last=False)


def memo(f):
    "Memoize function f, whose args must all be hashable."
    cache = BoundedLRU(100000)

    def fmemo(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]

    fmemo.__name__ = f.__name__ + "_memo"
    fmemo.cache = cache
    return fmemo


@memo
def _segment1(text):
    "Return a list of words that is the most probable segmentation of text."
    if not text:
        return []
    else:
        candidates = ([first] + _segment1(rest) for (first, rest) in splits(text, 1))
        return max(candidates, key=Pwords)


def cPword(word, prev):
    "Conditional probability of word, given previous word."
    bigram = prev + " " + word
    if P2w(bigram) > 0 and P1w(prev) > 0:
        return P2w(bigram) / P1w(prev)
    else:  # Average the back-off value and zero.
        return P1w(word) / 2
        # Smoothing destroys segmentation - not sure why.
        return P1w_smooth(word) / 2


def Pwords2(words, prev="<s>"):
    "The probability of a sequence of words, using bigram data, given prev word."
    # print([(w, (prev if (i == 0) else words[i-1]) )
    #                for (i, w) in enumerate(words)])
    return product(
        cPword(w, (prev if (i == 0) else words[i - 1])) for (i, w) in enumerate(words)
    )


def log_cPword(word, prev):
    "Conditional probability of word, given previous word."
    bigram = prev + " " + word
    if P2w(bigram) > 0 and P1w(prev) > 0:
        return math.log10(P2w(bigram) / P1w(prev))
    else:  # Average the back-off value and zero.
        return math.log10(P1w_smooth(word) / 2)


def log_prob_words2(words, prev="<s>"):
    "The probability of a sequence of words, using bigram data, given prev word."
    # print([(w, (prev if (i == 0) else words[i-1]) )
    #                for (i, w) in enumerate(words)])
    return sum(
        log_cPword(w, (prev if (i == 0) else words[i - 1]))
        for (i, w) in enumerate(words)
    )


@memo
def _segment2(text, prev="<s>"):
    "Return best segmentation of text; use bigram data."
    if not text:
        return []
    else:
        candidates = (
            [first] + _segment2(rest, first) for (first, rest) in splits(text, 1)
        )
        return max(candidates, key=lambda words: Pwords2(words, prev))


def segment_text(text):
    text = text.replace(" ", "")
    text = text.replace("'", "")
    # FIXME: hyphen too?
    return ", ".join([" ".join(ws.segment(t)) for t in text.split(",")])


# @memo
def _correct_word_or_segment(word, use_fine_distance):
    # Unigrams has way too many things that are readily confused as real words.
    # I mostly want "scrabble" words and a lot of these iffy unigrams appear quite often.
    top_word = word
    if word in WORDS:
        return [word]

    choices = []

    top_word_list = _correct_word_topn(word, 1)
    choices.append(top_word_list)
    seg_list = ws.segment(word)
    if len(seg_list) > 1:
        seg_list = [_correct_word_topn(w, 1)[0] for w in seg_list]
        choices.append(seg_list)
    # high score first
    choices.sort(key=log_prob_words2, reverse=True)
    return [" ".join(wl) for wl in choices]


# Combine spelling and segmenting, return highest scoring.
def correct_word(word):
    # FIXME: hack on fine distance
    return _correct_word_or_segment(word, use_fine_distance)[0]


# Sometimes I think that matching ' will slurp up contractions and help, but
# the data does not seem to bear that out.
# word_re = re.compile("[a-zA-Z]+['a-zA-Z][a-zA-Z]+")
word_re = re.compile("[a-zA-Z]+")


def tokenize(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return word_re.findall(text.lower())


def misspellings(text):
    return [t for t in tokenize(text) if t not in WORDS]


class BgProc(object):
    @staticmethod
    def _handler(f, in_queue, out_queue):
        while True:
            try:
                args, kargs = in_queue.get()
            except KeyboardInterrupt:
                return

            if kargs.get("__quit"):
                return

            try:
                r = f(*args, **kargs)
                out_queue.put((r, None))
            except Exception as e:
                out_queue.put((None, e))

    @staticmethod
    def fbody(*args, **kargs):
        raise Exception("unimplemented")

    def __init__(self):
        self.send_q = Queue()
        self.recv_q = Queue()

    def start(self):
        self.proc = Process(
            target=self._handler,
            args=(
                self.fbody,
                self.send_q,
                self.recv_q,
            ),
        )
        self.proc.start()

    def shutdown(self):
        self.send_q.put(((), {"__quit": True}))
        self.proc.join()

    def _put(self, *args, **kargs):
        self.send_q.put((args, kargs))

    def _get(self):
        (result, error) = self.recv_q.get()
        if error:
            raise Exception("spelling failed", str(error))
        return result

    def _call_async(self, *args, **kargs):
        self._put(*args, **kargs)
        return self._get

    def __del__(self):
        self.proc.kill()
        self.proc.join()


class SpellChecker(BgProc):
    fbody = staticmethod(correct_text)

    def correct_spelling_async(self, text):
        return self._call_async(text)


class Segmenter(BgProc):
    fbody = staticmethod(segment_text)

    def segment_async(self, text):
        return self._call_async(text)
