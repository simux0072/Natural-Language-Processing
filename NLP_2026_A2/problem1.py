import collections
import re

import matplotlib.ticker as ticker
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import brown, indian
from nltk.stem import WordNetLemmatizer

nltk.download("indian")
nltk.download("brown")

nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

lemmatizer = WordNetLemmatizer()


def is_word(token):
    return bool(re.search(r"[A-Za-z]", token))


def compute_stats(sentences, label):
    all_tokens = [tok for sent in sentences for tok in sent]
    word_tokens = [tok for tok in all_tokens if is_word(tok)]

    n_tokens = len(all_tokens)
    n_types = len(set(all_tokens))
    n_words = len(word_tokens)
    words_per_sent = [sum(1 for tok in sent if is_word(tok)) for sent in sentences]
    avg_words_sent = sum(words_per_sent) / len(words_per_sent) if sentences else 0

    avg_word_len = sum(len(w) for w in word_tokens) / n_words if n_words else 0
    lemmas = {lemmatizer.lemmatize(w.lower()) for w in word_tokens}
    n_lemmas = len(lemmas)

    freq_dist = collections.Counter(all_tokens)
    # An LLM was used to display metrics in a 'nice' way (I suck at UI Design)
    print(f"Statistics for: {label}")
    print(f"(i)   Tokens                    : {n_tokens:>10,}")
    print(f"(ii)  Types                     : {n_types:>10,}")
    print(f"(iii) Words (alphabetic tokens) : {n_words:>10,}")
    print(f"(iv)  Avg words / sentence      : {avg_words_sent:>10.2f}")
    print(f"(v)   Avg word length (chars)   : {avg_word_len:>10.2f}")
    print(f"(vi)  Lemmas (unique)           : {n_lemmas:>10,}")

    return dict(
        tokens=n_tokens,
        types=n_types,
        words=n_words,
        avg_words_per_sent=avg_words_sent,
        avg_word_length=avg_word_len,
        lemmas=n_lemmas,
        freq_dist=freq_dist,
    )


def sorted_freq_list(freq_dist):
    return freq_dist.most_common()


def top_pos_tags(sentences, n: int = 10):
    all_tokens = [tok for sent in sentences for tok in sent]
    tagged = nltk.pos_tag(all_tokens)
    tag_counts = collections.Counter(tag for _, tag in tagged)
    return tag_counts.most_common(n)


def plot_frequency_curves(
    datasets,
    title_prefix,
    filename_prefix,
):
    fig_lin, ax_lin = plt.subplots(figsize=(10, 5))
    fig_log, ax_log = plt.subplots(figsize=(10, 5))

    for label, freq_dist in datasets.items():
        freqs = [count for _, count in freq_dist.most_common()]
        ranks = range(1, len(freqs) + 1)

        ax_lin.plot(ranks, freqs, label=label, linewidth=1.2)
        ax_log.plot(ranks, freqs, label=label, linewidth=1.2)

    ax_lin.set_title(f"{title_prefix} – Frequency Curve (linear axes)")
    ax_lin.set_xlabel("Rank (position in frequency list)")
    ax_lin.set_ylabel("Frequency")
    ax_lin.legend()
    ax_lin.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_lin.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig_lin.tight_layout()
    lin_path = f"{filename_prefix}_linear.png"
    fig_lin.savefig(lin_path, dpi=150)
    print(f"\n  Saved: {lin_path}")

    ax_log.set_title(f"{title_prefix} – Frequency Curve (log-log axes)")
    ax_log.set_xlabel("Rank (log scale)")
    ax_log.set_ylabel("Frequency (log scale)")
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.legend()
    fig_log.tight_layout()
    log_path = f"{filename_prefix}_loglog.png"
    fig_log.savefig(log_path, dpi=150)
    print(f"  Saved: {log_path}")

    plt.close(fig_lin)
    plt.close(fig_log)


# An LLM was used to display metrics in a 'nice' way (I suck at UI Design)
def print_freq_list(freq_list, label, top_n: int = 30):
    print(f"\n  Top {top_n} tokens by frequency – {label}")
    print(f"  {'Rank':>5}  {'Token':<25} {'Frequency':>10}")
    print(f"  {'-'*5}  {'-'*25} {'-'*10}")
    for rank, (token, freq) in enumerate(freq_list[:top_n], start=1):
        print(f"  {rank:>5}  {token:<25} {freq:>10,}")


# An LLM was used to display metrics in a 'nice' way (I suck at UI Design)
def print_pos_tags(top_tags, label):
    print(f"\n  Top {len(top_tags)} POS tags – {label}")
    print(f"  {'Rank':>5}  {'Tag':<15} {'Count':>10}")
    print(f"  {'-'*5}  {'-'*15} {'-'*10}")
    for rank, (tag, count) in enumerate(top_tags, start=1):
        print(f"  {rank:>5}  {tag:<15} {count:>10,}")


print("BROWN CORPUS")

BROWN_GENRES = ("news", "romance")

brown_all = brown.sents()
brown_genre = {g: brown.sents(categories=g) for g in BROWN_GENRES}

brown_stats_all = compute_stats(list(brown_all), "Brown – full corpus")
brown_stats_genre = {
    g: compute_stats(list(brown_genre[g]), f"Brown – genre: {g}") for g in BROWN_GENRES
}

brown_freq_all = sorted_freq_list(brown_stats_all["freq_dist"])
print_freq_list(brown_freq_all, "Brown – full corpus")

for g in BROWN_GENRES:
    freq_list = sorted_freq_list(brown_stats_genre[g]["freq_dist"])
    print_freq_list(freq_list, f"Brown – genre: {g}")

print("POS Tagging (Brown corpus)")

brown_pos_all = top_pos_tags(list(brown.sents()))
print_pos_tags(brown_pos_all, "Brown – full corpus")

for g in BROWN_GENRES:
    pos_tags = top_pos_tags(list(brown.sents(categories=g)))
    print_pos_tags(pos_tags, f"Brown – genre: {g}")

print("Frequency plots (Brown)")

brown_plot_data = {"Full corpus": brown_stats_all["freq_dist"]}
for g in BROWN_GENRES:
    brown_plot_data[f"Genre: {g}"] = brown_stats_genre[g]["freq_dist"]

plot_frequency_curves(
    datasets=brown_plot_data,
    title_prefix="Brown Corpus",
    filename_prefix="brown",
)

INDIAN_FILES = {
    "Hindi": "hindi.pos",
    "Telugu": "telugu.pos",
}


def load_indian_sentences(fileids):
    try:
        sents = list(indian.sents(fileids))
    except Exception:
        sents = [list(indian.words(fileids))]
    return sents


indian_data = {lang: load_indian_sentences(fid) for lang, fid in INDIAN_FILES.items()}

all_indian_sents = [sent for sents in indian_data.values() for sent in sents]

indian_stats_all = compute_stats(all_indian_sents, "Indian – full corpus")
indian_stats_lang = {
    lang: compute_stats(sents, f"Indian – {lang}")
    for lang, sents in indian_data.items()
}

indian_freq_all = sorted_freq_list(indian_stats_all["freq_dist"])
print_freq_list(indian_freq_all, "Indian – full corpus")

for lang in INDIAN_FILES:
    freq_list = sorted_freq_list(indian_stats_lang[lang]["freq_dist"])
    print_freq_list(freq_list, f"Indian – {lang}")

print("POS Tagging (Indian corpus)")

indian_pos_all = top_pos_tags(all_indian_sents)
print_pos_tags(indian_pos_all, "Indian – full corpus")

for lang, sents in indian_data.items():
    pos_tags = top_pos_tags(sents)
    print_pos_tags(pos_tags, f"Indian – {lang}")

print("\nPOS tags in the Indian corpus:")
for lang, fid in INDIAN_FILES.items():
    tagged_words = indian.tagged_words(fid)
    native_tag_counts = collections.Counter(tag for _, tag in tagged_words)
    print(f"\nTop 10 native tags – {lang}")
    print(f"{'Rank':>5}  {'Tag':<20} {'Count':>10}")
    print(f"{'-'*5}  {'-'*20} {'-'*10}")
    for rank, (tag, count) in enumerate(native_tag_counts.most_common(10), start=1):
        print(f"{rank:>5}  {tag:<20} {count:>10,}")

print("Frequency plots (Indian)")

indian_plot_data = {"Full corpus": indian_stats_all["freq_dist"]}
for lang in INDIAN_FILES:
    indian_plot_data[lang] = indian_stats_lang[lang]["freq_dist"]

plot_frequency_curves(
    datasets=indian_plot_data,
    title_prefix="Indian Language Corpus",
    filename_prefix="indian",
)
