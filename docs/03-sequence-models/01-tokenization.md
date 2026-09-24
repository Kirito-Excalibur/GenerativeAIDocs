# Tokenization

> **Summary** — Before a language model sees text, the text must become integers. Tokenization
> decides the vocabulary, the sequence length, the cost per request, and a surprising number of
> model failures (arithmetic errors, the "strawberry has two r's" class of bug, poor performance in
> non-English languages). This page walks BPE through a complete merge-by-hand example, compares
> the three main algorithms, and catalogues the failure modes.

**Prerequisites**: → [Autoregressive models](../02-classical-models/01-autoregressive-models.md) · **Next**: → [RNNs, LSTMs & GRUs](02-rnn-lstm-gru.md)

---

## 1. The design space

```
  "unbelievable"
       │
       ├─ CHARACTER:  u|n|b|e|l|i|e|v|a|b|l|e        (12 tokens, vocab ~100)
       │              ✅ tiny vocab, no OOV ever
       │              ❌ long sequences → quadratic attention cost explodes
       │
       ├─ SUBWORD:    un|bel|iev|able                (4 tokens, vocab ~50k)
       │              ✅ the sweet spot — morphology captured, no OOV
       │
       └─ WORD:       unbelievable                   (1 token, vocab ~10⁶)
                      ✅ short sequences
                      ❌ huge embedding table, OOV for anything unseen
```

| | Character | **Subword** | Word |
|---|---|---|---|
| Vocabulary size | ~100–256 | **32k–256k** | 100k–1M+ |
| Sequence length (1000 chars) | 1000 | **~250** | ~170 |
| Out-of-vocabulary | impossible | impossible | frequent |
| Morphology | must be learned from scratch | partially captured | none |
| Embedding params ($d{=}4096$) | 1 M | **200 M** | 4 B |

> [!TIP]
> **Why subword won.** Sequence length enters attention cost quadratically, so character-level is
> ruinously expensive for long documents. Word-level has an OOV problem that no amount of data
> fixes (proper nouns, typos, code identifiers, new words). Subword units give you short sequences
> *and* the ability to spell out anything unseen, one piece at a time.

**The compression ratio is the number to know**: English text averages **~4 characters per
token** for a 50k-vocabulary BPE tokenizer. Code averages ~3.5. Non-Latin scripts are much worse
(§6).

---

## 2. Byte-Pair Encoding, worked completely by hand

That compression ratio is a property of one specific algorithm: byte-pair encoding. Rather than take it on faith, it's worth running BPE by hand on a small corpus and watching the vocabulary build up merge by merge.

BPE starts from characters and **greedily merges the most frequent adjacent pair**, repeatedly.

**Training corpus** (with word frequencies):

```
  "low"      × 5
  "lower"    × 2
  "newest"   × 6
  "widest"   × 3
```

**Initialize** — split into characters, with `</w>` marking a word boundary:

```
  l o w </w>          × 5
  l o w e r </w>      × 2
  n e w e s t </w>    × 6
  w i d e s t </w>    × 3
```

Initial vocabulary: `{l, o, w, e, r, n, s, t, i, d, </w>}` — 11 symbols.

### 🔢 Merge 1

Count every adjacent pair, weighted by word frequency:

| Pair | Occurrences | Count |
|---|---|---|
| `e s` | newest(6) + widest(3) | **9** ← winner |
| `s t` | newest(6) + widest(3) | 9 |
| `l o` | low(5) + lower(2) | 7 |
| `o w` | low(5) + lower(2) | 7 |
| `t </w>` | newest(6) + widest(3) | 9 |
| `w </w>` | low(5) | 5 |

Several pairs tie at 9; take `e s` (ties broken by first occurrence).
**Merge rule 1: `e s → es`**

```
  l o w </w>          × 5
  l o w e r </w>      × 2
  n e w es t </w>     × 6
  w i d es t </w>     × 3
```

### 🔢 Merge 2

| Pair | Count |
|---|---|
| `es t` | 6 + 3 = **9** ← winner |
| `l o` | 7 |
| `o w` | 7 |
| `t </w>` | 9 (tie) |

**Merge rule 2: `es t → est`**

```
  l o w </w>          × 5
  l o w e r </w>      × 2
  n e w est </w>      × 6
  w i d est </w>      × 3
```

### 🔢 Merge 3

| Pair | Count |
|---|---|
| `est </w>` | 6 + 3 = **9** ← winner |
| `l o` | 7 |
| `o w` | 7 |

**Merge rule 3: `est </w> → est</w>`**

### 🔢 Merges 4–6

| Step | Winning pair | Count | Result |
|---|---|---|---|
| 4 | `l o` | 7 | `lo` |
| 5 | `lo w` | 7 | `low` |
| 6 | `n e` | 6 | `ne` |

**Final state after 6 merges:**

```
  low </w>            × 5
  low e r </w>        × 2
  ne w est</w>        × 6
  w i d est</w>       × 3
```

**Learned merge rules (in order — the order is the algorithm):**

```
  1. e   s      → es
  2. es  t      → est
  3. est </w>   → est</w>
  4. l   o      → lo
  5. lo  w      → low
  6. n   e      → ne
```

### Encoding new text

To tokenize an unseen word, apply the merge rules **in the order they were learned**:

`"lowest"` → `l o w e s t </w>`
→ rule 1: `l o w es t </w>`
→ rule 2: `l o w est </w>`
→ rule 3: `l o w est</w>`
→ rule 4: `lo w est</w>`
→ rule 5: `low est</w>`
→ **`["low", "est</w>"]`** — 2 tokens.

> [!TIP]
> Notice what just happened: `"lowest"` never appeared in training, yet it decomposed into two
> meaningful morphemes. That generalization is why BPE works.

A complete, correct BPE trainer in 25 lines:

```python
from collections import Counter, defaultdict

def train_bpe(word_freqs, num_merges):
    # word_freqs: {"low": 5, "lower": 2, ...}
    vocab = {" ".join(list(w)) + " </w>": f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in vocab.items():
            syms = word.split()
            for i in range(len(syms) - 1):
                pairs[(syms[i], syms[i+1])] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        bigram, joined = " ".join(best), "".join(best)
        vocab = {w.replace(bigram, joined): f for w, f in vocab.items()}
    return merges

def encode(word, merges):
    syms = list(word) + ["</w>"]
    for a, b in merges:                    # apply IN LEARNED ORDER
        i = 0
        while i < len(syms) - 1:
            if syms[i] == a and syms[i+1] == b:
                syms[i:i+2] = [a + b]
            else:
                i += 1
    return syms
```

---

## 3. The three algorithms

That worked example used BPE's own merge rule, but it isn't the only way to build a subword vocabulary — the two main alternatives make a different call about what "best segmentation" even means.

| | **BPE** | **WordPiece** | **Unigram LM** |
|---|---|---|---|
| Direction | bottom-up (merge) | bottom-up (merge) | top-down (prune) |
| Merge criterion | highest **frequency** | highest **likelihood gain** | — |
| Selection score | $\text{count}(ab)$ | $\dfrac{\text{count}(ab)}{\text{count}(a)\cdot\text{count}(b)}$ | remove tokens that cost least likelihood |
| Segmentation | deterministic, greedy | deterministic, longest-match | **probabilistic** — Viterbi over a token LM |
| Subword regularization | ❌ (BPE-dropout hacks it in) | ❌ | ✅ native, sample different segmentations |
| Used by | GPT-2/3/4, LLaMA, Mistral, most LLMs | BERT, DistilBERT, ELECTRA | T5, ALBERT, XLNet, many multilingual models |

> [!TIP]
> **WordPiece's criterion, explained.** BPE merges `th` because "th" is common. But "t" and "h" are
> *both* individually very common, so their co-occurrence is not surprising. WordPiece divides by the
> individual frequencies, effectively using pointwise mutual information. It prefers merges where the
> pair appears together *more than chance would predict*. In practice the two produce similar
> vocabularies.

> [!TIP]
> **Unigram's difference is more fundamental.** It starts with a large candidate vocabulary and
> *removes* tokens, keeping those whose removal hurts corpus likelihood most. Critically, it keeps a
> probability for each token, so a word has *many* possible segmentations with different
> probabilities:

```
  "unbelievable"  →  un|bel|iev|able      p = 0.34
                  →  unbeliev|able        p = 0.28
                  →  un|believable        p = 0.21
                  →  u|n|b|e|l|i|e|v|able p = 0.001
```

**Subword regularization** samples a different segmentation each epoch, acting as data
augmentation. It measurably improves robustness, especially for low-resource languages and noisy
text.

---

## 4. Byte-level BPE: the trick that removes the last OOV

Whichever algorithm builds the vocabulary, there's still a gap at the very bottom: some input byte sequences are so rare — an emoji, a typo, raw binary — that no learned merge covers them. Byte-level BPE closes that gap completely, at a cost.

> [!WARNING]
> Character-level BPE has an unsolved edge case: what about a character that never appeared in
> training — a rare CJK ideograph, an emoji, a corrupted byte?

**GPT-2's answer**: operate on **UTF-8 bytes**, not characters. Base vocabulary = 256 byte values.
Every possible string is representable, because every string *is* bytes.

```
  "héllo"  ──UTF-8──►  [104, 195, 169, 108, 108, 111]
                             └──┬──┘
                          "é" is 2 bytes

  then run BPE over the byte sequence
```

**The cost**: a character outside ASCII takes 2–4 bytes, so worst case it becomes 2–4 tokens
before any merges apply. **The benefit**: genuinely zero OOV, ever. Used by GPT-2 onward, LLaMA,
and essentially every modern model.

> [!WARNING]
> **The pre-tokenization regex matters more than people expect.** GPT-2 splits text with a regex
> before BPE, so merges never cross word boundaries:

```python
pat = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

Note `| ?\p{L}+`: the leading space is attached to the *following* word. So `" cat"` is one token
and `"cat"` is a different one. This has real consequences:

In GPT-2's tokenizer: `" the"` = token 262, `"the"` = token 1169, `"The"` = token 464. **Three
different embeddings for the same word.** A prompt ending in a trailing space therefore forces the
model into the rarer, space-less continuation tokens and measurably degrades output. This is the
mechanism behind the common advice "don't end your prompt with a space."

> [!WARNING]
> **Number tokenization** is the other regex-driven trap. GPT-2 lets BPE merge digit runs by
> frequency, so the chunk boundaries have nothing to do with place value: `"1234"` becomes
> `"12"+"34"` but `"5678"` becomes `" 5"+"678"`. Arithmetic then operates over inconsistent units.
> **Two modern fixes**: split every digit individually (LLaMA-1/2, via SentencePiece), or cap digit
> chunks at 3 characters so they align with thousands (GPT-4's `cl100k` and LLaMA-3, where
> `"1234567"` → `"123"+"456"+"7"`). Both measurably improve arithmetic.

---

## 5. Special tokens

Digit-splitting and leading-space quirks are both about how ordinary text gets carved into pieces. A separate category of token doesn't come from text at all — special tokens the model itself relies on for structure — and they bring their own, sharper failure mode.

| Token | Purpose |
|---|---|
| `<|endoftext|>` / `</s>` | document separator; also the stop signal at generation time |
| `<pad>` | fill short sequences in a batch (masked out of the loss) |
| `<unk>` | unknown — **absent from byte-level tokenizers by construction** |
| `<|im_start|>`, `<|im_end|>` | chat turn delimiters |
| `<s>`, `[CLS]`, `[SEP]` | sequence start / classification / separator (BERT-style) |
| `<fim_prefix/suffix/middle>` | fill-in-the-middle, for code completion |

> [!WARNING]
> **Special tokens are a security boundary.** If a user's text can contain the literal string
> `<|im_start|>system`, and your tokenizer encodes it as the *special token* rather than as ordinary
> text, the user has just injected a system message. **Always encode untrusted input with special
> token parsing disabled**:

```python
tokenizer.encode(user_text, allowed_special=set())   # tiktoken: raises on special tokens
```

→ [Security](../08-safety-and-ethics/02-security.md)

---

## 6. The multilingual tax

Special-token injection is one specific way the tokenizer can hurt you. A more pervasive, structural cost shows up for anyone not writing in English, baked into which languages the merge rules were learned from.

Tokens required for the same semantic content (approximate, GPT-4-class tokenizer):

| Language | Chars/token | Tokens for the same paragraph | Relative cost |
|---|---|---|---|
| English | 4.0 | 100 | 1.0× |
| Spanish / French / German | 3.3 | ~120 | 1.2× |
| Russian | 2.5 | ~160 | 1.6× |
| Chinese | 1.5 | ~170 | 1.7× |
| Hindi (Devanagari) | 1.0 | ~250 | 2.5× |
| Burmese, Amharic, Telugu | 0.5–0.8 | ~400+ | 4×+ |

**This is not a cosmetic issue.** It means, for identical content:

1. **You pay 2–4× more** per API call (billing is per token).
2. **You get 2–4× less** usable context window.
3. **Quality is worse** — the model sees a longer, more fragmented sequence for the same meaning,
   and each token carries less information.
4. **Latency is 2–4× higher** — generation is per-token.

> [!TIP]
> The cause is the training-data distribution: merges are learned from a predominantly English
> corpus, so English words become single tokens while Telugu words are spelled out byte by byte.
> Multilingual-first tokenizers (mT5, NLLB, Gemma's 256k vocabulary, Aya) deliberately oversample
> low-resource languages when learning merges, which narrows the gap substantially.

---

## 7. Failure modes caused by tokenization

The multilingual tax is really just one instance of a broader pattern: whatever the tokenizer didn't see enough of during its own training becomes a blind spot for the model built on top of it. Cataloguing where that blind spot actually shows up is the point of this section.

### (a) Character-level tasks

> "How many r's in 'strawberry'?"

The model sees `["str", "aw", "berry"]` — **the letters are not individually visible.** Asking it
to count characters is like asking someone to count the letters in a word they only ever heard
spoken. Models can learn this indirectly (they've seen spelling discussed in text), but it is
inference from memorized facts, not perception.

Same root cause: reversing strings, counting letters, rhyming, pig latin, acrostics, and
character-level ciphers.

### (b) Arithmetic

```
  "1234 + 5678"  →  ["12", "34", " +", " 5", "678"]        (GPT-2 tokenizer, verified with tiktoken)
```

The digits are not aligned to place value. The model must learn arithmetic over *inconsistent
groupings* — as if you had to add numbers written in a base that changes per number.

**Fixes that measurably work**: single-digit tokenization (now standard), right-to-left digit
grouping in threes, and explicitly reversed-digit output formats in fine-tuning data.

### (c) Glitch tokens

Some tokens exist in the vocabulary but appear ~never in the training data, because the
tokenizer was trained on a corpus that included data (e.g. scraped Reddit usernames) later filtered
out of the LM training set. Their embeddings stay near their random initialization.

The famous GPT-2/3 example: `SolidGoldMagikarp` (a Reddit username). Prompting with it produced
bizarre, off-distribution behaviour — the model was being asked to condition on an embedding it
had essentially never trained on.

> [!TIP]
> **The lesson**: tokenizer training set and model training set should be the same corpus.

### (d) The trailing-space problem

Because of the leading-space convention (§4): a prompt ending with `"The answer is "` forces the
next token to be the space-less variant, which is rarer and worse-modelled. Always end prompts at
a natural token boundary — `"The answer is"` — and let the model emit the space.

---

## 8. Practical numbers

All of the failure modes above are qualitative warnings. Turning "tokenization matters" into an actual planning number — how many tokens will this cost, how much context does that leave — is what the rest of this page is for.

**Vocabulary sizes**:

| Model family | Vocab | Notes |
|---|---|---|
| GPT-2 | 50,257 | byte-level BPE |
| GPT-3 | 50,257 | same tokenizer |
| GPT-4 (cl100k) | 100,277 | better multilingual & code |
| LLaMA 1/2 | 32,000 | SentencePiece BPE, digit splitting |
| LLaMA 3 | 128,256 | much better compression |
| Gemma | 256,000 | multilingual-first |
| Qwen | ~152k | strong Chinese coverage |

**The embedding-table trade-off.** With $d = 4096$:

| Vocab | Embedding params | Share of a 7B model |
|---|---|---|
| 32k | 131 M (×2 if untied) | 1.9% |
| 128k | 524 M (×2 if untied) | 7.5% |
| 256k | 1.05 B (×2 if untied) | 15% |

A bigger vocabulary buys shorter sequences (cheaper attention, more content per context window) but
costs embedding parameters and makes the final softmax more expensive. Empirically **vocabulary
size should scale with model size** — Tao et al. (2024) find compute-optimal vocabulary grows
sub-linearly with $N$, and that most models before 2024 were *under*-vocabularied.

**Always measure, never assume:**

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
text = open("sample.txt").read()
toks = enc.encode(text)
print(f"{len(text)} chars -> {len(toks)} tokens ({len(text)/len(toks):.2f} chars/token)")
print([enc.decode([t]) for t in toks[:20]])   # look at the actual pieces
```

Looking at the actual token pieces for your domain is a five-minute check that routinely explains
mysterious quality problems.

---

## 9. Exercises

**Problem 1 — BPE merge, a new corpus.** Corpus: "low" ×3, "new" ×5, "wide" ×2. Initialize as
characters with `</w>` boundary markers (as in §2), count all adjacent pairs, and identify the
first merge.

<details markdown="1"><summary>Solution</summary>

Initial split: `l o w </w>` ×3, `n e w </w>` ×5, `w i d e </w>` ×2.

Pair counts: $(l,o){=}3$, $(o,w){=}3$, $(w,{<}/w{>}){=}3{+}5{=}8$, $(n,e){=}5$, $(e,w){=}5$,
$(w,i){=}2$, $(i,d){=}2$, $(d,e){=}2$, $(e,{<}/w{>}){=}2$.

**First merge: `w </w>` → `w</w>`**, with count **8** (the unique highest — it wins because both
"low" and "new" end in "w" immediately before the boundary, and their frequencies add:
$3{+}5{=}8$, beating any single-word-only pair). Note this differs from §2's worked example,
where `e s` won — here it's a *boundary* merge that wins first because two different words share
the same word-final bigram.

</details>

**Problem 2 — the leading-space trap, applied.** Using §4's discussion, predict what happens (in
terms of token IDs / embeddings, not exact numbers) if a fine-tuning dataset is built by
string-concatenating `prompt + "\n" + response` where `response` sometimes starts with a leading
space (inconsistently, depending on how the data was scraped). What symptom would you expect at
inference time?

<details markdown="1"><summary>Solution</summary>

Per §4, `"the"` and `" the"` (with a leading space) are **different tokens** with different
embeddings under GPT-2-style byte-level BPE (the space attaches to the *following* word by
construction of the pre-tokenization regex). If training examples inconsistently include or omit
that leading space, the model sees the *same underlying word* split across two different token
IDs depending on an accident of data formatting — diluting the training signal for whichever
concept that word represents, since gradient updates for `"the"` and `" the"` don't share
weights.

At inference: the model may behave inconsistently depending on whether its own prior context
happens to end in a way that makes it emit the space-prefixed or space-less variant next,
producing subtly different completions for what should be equivalent prompts — the same class of
problem §4 describes for prompts ending in a trailing space, just introduced through noisy
training data instead of a hand-written prompt.

</details>

**Problem 3 — multilingual cost, concretely.** Using §6's per-language chars/token ratios,
estimate how many tokens are needed to encode a 2,000-character document in English (4.0
chars/token) versus the same content translated into Hindi (1.0 chars/token, per §6's table,
though the *character count* of a translation isn't identical — assume for this exercise it stays
2,000 characters for simplicity). What's the cost multiple in tokens, and name two concrete
consequences from §6 beyond just "more tokens."

<details markdown="1"><summary>Solution</summary>

English: $2000/4.0 = 500$ tokens. Hindi: $2000/1.0=2000$ tokens. **4× more tokens** for the same
character count — matching §6's "2.5×" relative-cost figure for Hindi in the table (the
discrepancy is because that table's ratio is computed against a matched-content baseline, not
raw character count, but the direction and rough magnitude agree).

Two concrete consequences beyond raw token count, both listed in §6: (1) the document now
consumes 4× more of the model's fixed context window, so less of a long Hindi document fits
before hitting the context limit than the equivalent English document would; (2) per-request
API cost is billed per token, so processing this document costs roughly 4× more in Hindi than in
English for identical semantic content — a direct, measurable "multilingual tax."

</details>

## 10. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Subword tokenization is the compromise: no OOV (unlike words), short sequences (unlike characters). |
| 2 | BPE = greedily merge the most frequent adjacent pair; apply merge rules in learned order at encode time. |
| 3 | WordPiece uses a PMI-like criterion; Unigram prunes a large vocabulary and supports probabilistic segmentation. |
| 4 | Byte-level BPE eliminates OOV entirely by operating on UTF-8 bytes. |
| 5 | The leading-space convention means `"the"` and `" the"` are different tokens — never end a prompt with a space. |
| 6 | Non-English text costs 2–4× more tokens: more money, less context, worse quality. |
| 7 | Character counting, arithmetic and rhyming fail largely *because of* tokenization, not reasoning. |
| 8 | Split digits individually; keep tokenizer and model training corpora aligned (glitch tokens). |
| 9 | English ≈ 4 chars/token; always measure on your own domain. |

---

## Further reading

- Sennrich et al., [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909) (2016) — BPE for NMT.
- Kudo, [*Subword Regularization*](https://arxiv.org/abs/1804.10959) (2018) — the Unigram LM model.
- Kudo & Richardson, [*SentencePiece*](https://arxiv.org/abs/1808.06226) (2018).
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — byte-level BPE.
- Tao et al., [*Scaling Laws with Vocabulary*](https://arxiv.org/abs/2407.13623) (2024).
- Ahia et al., [*Do All Languages Cost the Same?*](https://arxiv.org/abs/2305.13707) (2023) — the multilingual tokenization tax.

**Next** → [RNNs, LSTMs & GRUs](02-rnn-lstm-gru.md)
