# Google Gemini was used to make the data easier to see (it fixed up the data view and added formatting to items
# because I am really bad at UIs and such...).

import collections
import math
import re

from nltk.corpus import brown

MIN_COUNT = 10
TOP_N = 20
BROWN100_FILE = "./brown_100.txt"  # Path to the brown_100 file.


def is_word(token):
    return bool(re.search(r"[A-Za-z]", token))


def tokenise_sentences(sentences):
    return [
        [tok.lower() for tok in sent if is_word(tok)]
        for sent in sentences
        if any(is_word(tok) for tok in sent)
    ]


def count_unigrams_and_bigrams(tokenised_sents):
    unigrams = collections.Counter()
    bigrams = collections.Counter()

    for sent in tokenised_sents:
        unigrams.update(sent)
        for w1, w2 in zip(sent, sent[1:]):
            bigrams[(w1, w2)] += 1

    N = sum(bigrams.values())
    return unigrams, bigrams, N


def compute_pmi(unigrams, bigrams, N, min_count=MIN_COUNT):
    # Build set of words that meet the frequency threshold
    frequent = {w for w, c in unigrams.items() if c >= min_count}

    pmi_scores = {}
    for (w1, w2), c_bigram in bigrams.items():
        if w1 not in frequent or w2 not in frequent:
            continue
        c_w1 = unigrams[w1]
        c_w2 = unigrams[w2]
        score = math.log2((c_bigram * N) / (c_w1 * c_w2))
        pmi_scores[(w1, w2)] = score

    return sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)


def compute_ppmi(pmi_sorted: list):
    ppmi_scores = [((w1, w2), max(0.0, score)) for (w1, w2), score in pmi_sorted]
    return sorted(ppmi_scores, key=lambda x: x[1], reverse=True)


def print_pmi_table(pairs, title, n=TOP_N):
    """Google Gemini Prettyfied-print the top-n and bottom-n entries from a PMI list."""
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")
    print(f"  {'Rank':>5}  {'w1':<18} {'w2':<18} {'PMI':>8}")
    print(f"  {'-'*5}  {'-'*18} {'-'*18} {'-'*8}")
    for rank, ((w1, w2), score) in enumerate(pairs[:n], 1):
        print(f"  {rank:>5}  {w1:<18} {w2:<18} {score:>8.4f}")
    if len(pairs) > n:
        print(f"\n  ... ({len(pairs)} pairs total) ...\n")
        print(f"  {'Rank':>5}  {'w1':<18} {'w2':<18} {'PMI':>8}")
        print(f"  {'-'*5}  {'-'*18} {'-'*18} {'-'*8}")
        for rank, ((w1, w2), score) in enumerate(pairs[-n:], len(pairs) - n + 1):
            print(f"  {rank:>5}  {w1:<18} {w2:<18} {score:>8.4f}")


print("\n" + "#" * 65)
print("  Step 1 – Full Brown Corpus")
print("#" * 65)

# Step 1 – load and tokenise
brown_sents = tokenise_sentences(brown.sents())
unigrams_b, bigrams_b, N_b = count_unigrams_and_bigrams(brown_sents)

print(f"\n  Corpus size  : {sum(unigrams_b.values()):>10,} word tokens")
print(f"  Vocabulary   : {len(unigrams_b):>10,} unique words")
print(f"  Bigram types : {len(bigrams_b):>10,} unique pairs")
print(f"  Bigram tokens: {N_b:>10,}")
print(
    f"  Words with freq >= {MIN_COUNT}: {sum(1 for c in unigrams_b.values() if c >= MIN_COUNT):,}"
)

# Step 2 – PMI
pmi_brown = compute_pmi(unigrams_b, bigrams_b, N_b, min_count=MIN_COUNT)

print_pmi_table(pmi_brown, "Top 20 PMI pairs – Full Brown corpus", n=TOP_N)
print()
print_pmi_table(
    list(reversed(pmi_brown)),
    "Bottom 20 PMI pairs – Full Brown corpus",
    n=TOP_N,
)

# The Brown corpus contains ~1.15 million tokens across 500 texts spanning 15 genres.
# Working on the full corpus gives pretty stable estimates. The MIN_COUNT=10 filter removes
# very rare words that would otherwise produce very extreme PMI values.

ppmi_brown = compute_ppmi(pmi_brown)

n_zero = sum(1 for _, s in ppmi_brown if s == 0.0)
n_positive = len(ppmi_brown) - n_zero

print(f"\n{'═'*65}")
print(f"  PPMI summary – Full Brown corpus")
print(f"{'═'*65}")
print(f"  Total qualifying pairs : {len(ppmi_brown):>8,}")
print(f"  Pairs with PPMI > 0    : {n_positive:>8,}")
print(f"  Pairs with PPMI = 0    : {n_zero:>8,}  (negative PMI clamped)")

print_pmi_table(ppmi_brown, "Top 20 PPMI pairs – Full Brown corpus", n=TOP_N)

print("\n" + "#" * 65)
print("  PART B – BROWN100 SNIPPET")
print("#" * 65)

MIN_COUNT_100 = 10  # could lower the limit so that more words would be evaluated more.

# The brown100 only contains the first 100 sentences of the Brown corpus.
# It is very small and thus reducing the MIN_COUNT=10 threshold to a lower value
# would retain more words (most words appear only once or twice in the 100 sentences).


def parse_brown100(filepath):
    sentences = []
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Remove <s> and </s> tags
            line = re.sub(r"</?s>", "", line).strip()
            tokens = [tok.lower() for tok in line.split() if is_word(tok)]
            if tokens:
                sentences.append(tokens)
    return sentences


brown100_sents = parse_brown100(BROWN100_FILE)
unigrams_100, bigrams_100, N_100 = count_unigrams_and_bigrams(brown100_sents)

print(f"\n  Corpus size  : {sum(unigrams_100.values()):>10,} word tokens")
print(f"  Vocabulary   : {len(unigrams_100):>10,} unique words")
print(f"  Bigram types : {len(bigrams_100):>10,} unique pairs")
print(f"  Bigram tokens: {N_100:>10,}")
print(
    f"  Words with freq >= {MIN_COUNT_100}: "
    f"{sum(1 for c in unigrams_100.values() if c >= MIN_COUNT_100):,}"
)

# PMI on brown100
pmi_100 = compute_pmi(unigrams_100, bigrams_100, N_100, min_count=MIN_COUNT_100)

print_pmi_table(
    pmi_100, f"Top 20 PMI pairs – brown100  (min_count={MIN_COUNT_100})", n=TOP_N
)
print()
print_pmi_table(
    list(reversed(pmi_100)),
    f"Bottom 20 PMI pairs – brown100  (min_count={MIN_COUNT_100})",
    n=TOP_N,
)

# PPMI on brown100
ppmi_100 = compute_ppmi(pmi_100)

n_zero_100 = sum(1 for _, s in ppmi_100 if s == 0.0)
n_positive_100 = len(ppmi_100) - n_zero_100

print(f"\n{'═'*65}")
print(f"  PPMI summary – brown100")
print(f"{'═'*65}")
print(f"  Total qualifying pairs : {len(ppmi_100):>8,}")
print(f"  Pairs with PPMI > 0    : {n_positive_100:>8,}")
print(f"  Pairs with PPMI = 0    : {n_zero_100:>8,}  (negative PMI clamped)")

print_pmi_table(
    ppmi_100, f"Top 20 PPMI pairs – brown100  (min_count={MIN_COUNT_100})", n=TOP_N
)
