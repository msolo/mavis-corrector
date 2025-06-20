import spell_checker
from spell_checker import (
    correct_text,
    tokenize,
    ws,
    _segment2,
)

_test_spelling_pairs = [
    # Generated misses from out data
    ("aa", "a"),
    ("aan", "can"),
    ("aas", "was"),
    ("accounts", "accounts"),
    ("algeria", "algeria"),
    ("aseed", "asked"),
    ("ber", "be"),
    ("ccan", "can"),
    ("checks", "checks"),
    ("confusing", "confusing"),
    ("cu", "cut"),
    ("dates", "dates"),
    ("deciding", "deciding"),
    ("delunciations", "denunciations"),
    ("depends", "depends"),
    ("died", "died"),
    ("earlier", "earlier"),
    ("eerrythhing", "everything"),
    ("email", "email"),
    ("email", "email"),
    ("ezaa", "ezra"),
    ("falls", "falls"),
    ("feels", "feels"),
    ("firt", "first"),
    ("flowers", "flowers"),
    ("fridge", "fridge"),
    ("gets", "gets"),
    ("gues", "guess"),
    ("guys", "guys"),
    ("having", "having"),
    ("including", "including"),
    ("insetrs", "inserts"),
    ("irt", "it"),
    ("ispec=tion", "inspection"),
    ("jad", "had"),
    ("kaaakhstan", "kazakhstan"),
    ("kidn", "kind"),
    ("knocl", "knock"),
    ("ko", "to"),
    ("langugges", "languages"),
    ("legs", "legs"),
    ("looked", "looked"),
    ("lrt", "let"),
    ("macintoch", "macintosh"),
    ("makes", "makes"),
    ("mentioned", "mentioned"),
    ("mhy", "my"),
    ("mortaages", "mortgages"),
    ("names", "names"),
    ("okkes", "makes"),
    ("online", "online"),
    ("oont", "wont"),
    ("oot", "not"),
    ("oow", "how"),
    ("osn", "on"),
    ("owt", "out"),
    ("passes", "passes"),
    ("pieps", "pipes"),
    ("pikle", "pile"),
    ("possitilities", "possibilities"),
    ("pyyout", "payout"),
    ("pyyout", "payout"),
    ("questions", "questions"),
    ("realized", "realized"),
    ("reallycouldn't", "really couldn't"),
    ("sattes", "states"),
    ("savings", "savings"),
    ("searches", "searches"),
    ("sleot", "slept"),
    ("sokks", "socks"),
    ("stitches", "stitches"),
    ("strethcing", "stretching"),
    ("terms", "terms"),
    ("tghe", "the"),
    ("th", "the"),
    ("thatand", "that and"),
    ("thefair", "the fair"),
    ("theyshoudl", "they should"),
    ("things", "things"),
    ("thre", "three"),
    ("tnm", "tim"),
    ("tye", "the"),
    ("tye", "the"),
    ("typing", "typing"),
    ("uch", "much"),
    ("varietis", "varieties"),
    ("vernont", "vermont"),
    ("weathervaene", "weathervane"),
    ("weathervaene", "weathervane"),
    ("wellit", "well it"),
    ("witnesess", "witnesses"),
    ("wss", "was"),
    ("yesi", "yes"),
]
_test_spelling_pairs.sort()

_test_spelling_expected_misses = {
    "cu",
    "eerrythhing",
    "gues",
    "ispec=tion",
    "reallycouldn't",
    "th",
    "thatand",
    "thefair",
    "thre",
    "tnm",
    "vernont",
}


def _test_spelling():
    misses = []
    for src, dst in _test_spelling_pairs:
        x = correct_text(src)
        if x != dst:
            misses.append((src, dst, x))
    return misses


def print_misses(misses, squelch_expected):
    for src, expected, result in misses:
        expected_miss = src in _test_spelling_expected_misses
        if result != expected:
            if expected_miss and squelch_expected:
                continue
            print(f"miss: {src} -> {result} expected: {expected}")
        elif expected_miss:
            print(f"unexpected hit: {src} -> {result} excpected: {expected}")
    print("misses:", len(misses), "total:", len(_test_spelling_pairs))


def test_spelling(squelch_expected=False):
    spell_checker.use_fine_distance = False
    misses = _test_spelling()
    print_misses(misses, squelch_expected)


def test_fine_spelling(squelch_expected=False):
    spell_checker.use_fine_distance = True
    fine_misses = _test_spelling()
    print_misses(fine_misses, squelch_expected)


def _test_segmenter(segmenter_f, tests, expected_misses):
    "Try segmenter on tests; report failures; return fraction correct."
    return sum(
        [
            _test_one_segment(segmenter_f, test, test in expected_misses)
            for test in tests
        ]
    ), len(tests)


def _test_one_segment(segmenter_f, test, expected_miss):
    words = tokenize(test)
    result = segmenter_f("".join(words))
    correct = result == words
    if not correct and not expected_miss:
        print(segmenter_f.__name__)
        print("expected", words)
        print("got     ", result)
    elif correct and expected_miss:
        print(segmenter_f.__name__)
        print("expected miss", words)
    return correct


_test_segmenter_lines = [
    x.strip()
    for x in (
        """A little knowledge is a dangerous thing
  A man who is his own lawyer has a fool for his client
  All work and no play makes Jack a dull boy
  Better to remain silent and be thought a fool that to speak and remove all doubt;
  Do unto others as you would have them do to you
  Early to bed and early to rise, makes a man healthy, wealthy and wise
  Fools rush in where angels fear to tread
  Genius is one percent inspiration, ninety-nine percent perspiration
  If you lie down with dogs, you will get up with fleas
  Lightning never strikes twice in the same place
  Power corrupts; absolute power corrupts absolutely
  Here today, gone tomorrow
  See no evil, hear no evil, speak no evil
  Sticks and stones may break my bones, but words will never hurt me
  Take care of the pence and the pounds will take care of themselves
  Take care of the sense and the sounds will take care of themselves
  The bigger they are, the harder they fall
  The grass is always greener on the other side of the fence
  The more things change, the more they stay the same
  Those who do not learn from history are doomed to repeat it
  choose spain
  speed of art
  small and insignificant
  large and insignificant
  far out in the uncharted backwaters of the unfashionable end of the western spiral arm of the galaxy lies a small unregarded yellow sun""".splitlines()
    )
    if x.strip()
]

_test_segmenter_expected_misses = set(
    [
        x.strip()
        for x in (
            """Fools rush in where angels fear to tread
  far out in the uncharted backwaters of the unfashionable end of the western spiral arm of the galaxy lies a small unregarded yellow sun""".splitlines()
        )
        if x.strip()
    ]
)


def test_segmenter():
    hits, total = _test_segmenter(
        ws.segment, _test_segmenter_lines, _test_segmenter_expected_misses
    )
    print("wordsegment misses:", total - hits, "total:", total)
    # hits, total = _test_segmenter(_segment1, _test_segmenter_lines)
    # print(hits)
    # This seems basically awful at this point.
    hits, total = _test_segmenter(
        _segment2, _test_segmenter_lines, _test_segmenter_expected_misses
    )
    print(hits)


def _test_all():
    test_spelling(squelch_expected=True)
    test_fine_spelling(squelch_expected=True)
    test_segmenter()


if __name__ == "__main__":
    _test_all()
