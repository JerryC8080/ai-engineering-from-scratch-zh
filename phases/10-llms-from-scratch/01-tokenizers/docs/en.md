# Tokenizers: BPE, WordPiece, SentencePiece

> The tokenizer is the front door of every language model - nothing gets in without passing through it first.

**Type:** Build
**Languages:** Python, Rust
**Prerequisites:** Phase 5 (NLP Foundations)
**Time:** ~90 minutes

## The Problem

You feed a sentence into GPT. What does the model actually see?

Not characters. Not words. Tokens.

The tokenizer decides how text gets sliced into pieces before a model ever touches it. That decision shapes everything downstream: vocabulary size, sequence length, out-of-vocabulary handling, multilingual support, arithmetic ability, even the cost of an API call (you pay per token).

A bad tokenizer wastes context window on redundant subwords. A good one compresses common patterns and gracefully handles rare words. The difference between "this model understands code" and "this model chokes on variable names" often comes down to tokenizer design.

If you skip this, you will not understand why GPT tokenizes " New" and "York" separately, why BERT handles "[UNK]" tokens, or why some models burn through your context window twice as fast on non-English text.

## The Concept

### Three Approaches to Splitting Text

```
Character-level:    "hello" -> ["h", "e", "l", "l", "o"]
Word-level:         "hello world" -> ["hello", "world"]
Subword-level:      "unhappiness" -> ["un", "happi", "ness"]
```

Each has tradeoffs:

| Approach | Vocabulary Size | Sequence Length | OOV Handling | Example |
|----------|----------------|-----------------|--------------|---------|
| Character | ~256 | Very long | None (all chars known) | GPT-1 early experiments |
| Word | 100K+ | Short | Poor (unknown words) | Classical NLP |
| Subword | 30K-100K | Medium | Good (decomposes unknowns) | GPT, BERT, LLaMA |

Subword tokenization won. Every modern LLM uses it. The question is which subword algorithm.

### BPE: Byte Pair Encoding

BPE starts with individual characters and repeatedly merges the most frequent adjacent pair. It is a greedy compression algorithm repurposed for tokenization.

Here is how it works on a tiny corpus:

```
Corpus: "hug hug hug pug pug bug"

Step 0 - Start with characters:
  h u g   h u g   h u g   p u g   p u g   b u g

Step 1 - Count all adjacent pairs:
  (h,u): 3   (u,g): 6   (g, ): 5   ( ,h): 2
  ( ,p): 2   (p,u): 2   ( ,b): 1

Step 2 - Merge most frequent pair (u,g) -> "ug":
  h ug   h ug   h ug   p ug   p ug   b ug

Step 3 - Recount pairs:
  (h,ug): 3   (ug, ): 5   ( ,h): 2   ( ,p): 2
  (p,ug): 2   ( ,b): 1

Step 4 - Merge most frequent (ug, ) -> "ug ":
  h "ug "  h "ug "  h "ug "  p "ug "  p "ug "  b ug

Step 5 - Continue until vocabulary size target reached...
```

The merge table becomes your tokenizer. To encode new text, apply merges in the same order they were learned.

```
BPE Merge Process (ASCII diagram):

Input text:  "unhappily"

Start:       u  n  h  a  p  p  i  l  y
              \/
Merge 1:     un h  a  p  p  i  l  y       (u+n -> un, if learned)
                    \  /
Merge 2:     un h  ap p  i  l  y           (a+p -> ap, if learned)
                    \  /
Merge 3:     un h  app   i  l  y           (ap+p -> app, if learned)
                 \  /
Merge 4:     un happ     i  l  y           (h+app -> happ, if learned)
                    \     /
Merge 5:     un happi    l  y              (app+i -> appi, if learned)
                      \  /
Merge 6:     un happi   ly                 (l+y -> ly, if learned)

Result:      ["un", "happi", "ly"]
```

### Byte-Level BPE (GPT-2, GPT-3, GPT-4)

Standard BPE operates on Unicode characters. Byte-level BPE operates on raw bytes (0-255). This gives you a base vocabulary of exactly 256, handles any language or encoding, and never produces an unknown token.

GPT-2 introduced this. The trick: map each byte to a visible Unicode character so the vocabulary stays human-readable. The byte 0x20 (space) becomes "G", 0x41 ('A') stays 'A', and so on.

tiktoken (OpenAI's tokenizer library) uses byte-level BPE with a vocabulary of ~100K tokens for GPT-4.

### WordPiece (BERT)

WordPiece is similar to BPE but picks merges differently. Instead of raw frequency, it maximizes the likelihood of the training data:

```
BPE merge criterion:      count(A, B)
WordPiece merge criterion: count(AB) / (count(A) * count(B))
```

WordPiece favors merges where the pair appears together more often than you would expect by chance. It also uses a "##" prefix for continuation tokens:

```
"unhappiness" -> ["un", "##happi", "##ness"]
```

The "##" tells you this piece continues a previous token rather than starting a new word.

### SentencePiece

SentencePiece treats the input as a raw stream of Unicode characters, including whitespace. It does not require pre-tokenized words. This makes it language-agnostic - it works on Chinese, Japanese, Thai, and other languages where word boundaries are not marked by spaces.

SentencePiece supports both BPE and Unigram algorithms. LLaMA, T5, and many multilingual models use SentencePiece.

The Unigram approach works in reverse compared to BPE: start with a large vocabulary and iteratively remove tokens that least affect the overall likelihood.

### How Tokenizer Choice Affects the Model

The tokenizer is not neutral. It bakes in assumptions:

**Vocabulary size tradeoff:**
- Larger vocabulary (100K+): shorter sequences, more parameters in the embedding layer
- Smaller vocabulary (30K): longer sequences, smaller embedding layer, better generalization to rare words

**Fertility (tokens per word):**
- English text in GPT-4: ~1.3 tokens per word
- Korean text in GPT-4: ~2-3 tokens per word
- Code: highly variable, depends on training data

**Downstream effects:**
- Arithmetic: "1234" tokenized as ["123", "4"] vs ["1", "234"] changes whether the model can learn digit-level operations
- Code: a tokenizer trained mostly on English text wastes tokens on Python indentation
- Multilingual: models that tokenize non-English text into many small pieces effectively have a shorter context window for those languages

## Build It

### Step 1: Basic BPE Tokenizer

We build a complete BPE tokenizer from scratch. The training loop counts pairs, finds the most frequent, merges it, and records the merge rule.

```python
from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def _get_pairs(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, text, num_merges):
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = 256 + i
            tokens = self._merge_pair(tokens, best_pair, new_token)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            merged_str = self.vocab[new_token]
            print(f"Merge {i + 1}: {best_pair} -> {new_token} = {merged_str}")

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for pair, new_token in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_token)
        return tokens

    def decode(self, tokens):
        byte_sequence = b"".join(self.vocab[t] for t in tokens)
        return byte_sequence.decode("utf-8", errors="replace")
```

### Step 2: Train and Test

```python
corpus = """The cat sat on the mat. The cat ate the rat.
The dog sat on the log. The dog ate the frog.
Natural language processing is the study of how computers
understand and generate human language."""

tokenizer = BPETokenizer()
tokenizer.train(corpus, num_merges=30)

test = "The cat sat on the mat."
encoded = tokenizer.encode(test)
decoded = tokenizer.decode(encoded)

print(f"\nOriginal: {test}")
print(f"Encoded:  {encoded}")
print(f"Decoded:  {decoded}")
print(f"Tokens:   {len(encoded)} (from {len(test.encode('utf-8'))} bytes)")
```

### Step 3: Compare With tiktoken

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "The cat sat on the mat."
tokens = enc.encode(text)
print(f"tiktoken tokens: {tokens}")
print(f"tiktoken decoded: {[enc.decode([t]) for t in tokens]}")
print(f"Token count: {len(tokens)}")

text2 = "unhappiness"
tokens2 = enc.encode(text2)
print(f"\n'{text2}' -> {[enc.decode([t]) for t in tokens2]}")
print(f"Token count: {len(tokens2)}")
```

tiktoken uses the same BPE algorithm, but trained on a massive corpus with 100K merges. The merge table is what makes it powerful, not the algorithm itself.

## Use It

### SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=1000,
    model_type="bpe"
)

sp = spm.SentencePieceProcessor()
sp.load("my_tokenizer.model")

tokens = sp.encode("The cat sat on the mat.", out_type=str)
print(tokens)
ids = sp.encode("The cat sat on the mat.")
print(ids)
print(sp.decode(ids))
```

### Hugging Face Tokenizers

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(vocab_size=1000, special_tokens=["<pad>", "<eos>", "<unk>"])
tokenizer.train(["corpus.txt"], trainer)

output = tokenizer.encode("The cat sat on the mat.")
print(output.tokens)
print(output.ids)
```

The Hugging Face `tokenizers` library is written in Rust under the hood. It trains BPE on gigabyte-scale corpora in seconds.

### Rust for Production Tokenization

When you need to tokenize millions of documents for pre-training, Python becomes the bottleneck. The Rust implementation in `code/bpe.rs` shows how the same algorithm runs 10-50x faster with zero-copy byte handling.

## Ship It

This lesson produces a skill for choosing and building tokenizers in LLM projects. See `outputs/skill-tokenizer.md`.

## Exercises

1. **Easy:** Modify the BPE tokenizer to print the vocabulary at each merge step. Observe how common English words get assembled piece by piece.
2. **Medium:** Add special tokens (`<pad>`, `<eos>`, `<unk>`) to the BPE tokenizer. Implement pre-tokenization that splits on whitespace before running BPE.
3. **Hard:** Implement the WordPiece merge criterion (likelihood-based instead of frequency-based). Compare the vocabularies produced by BPE vs WordPiece on the same corpus.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Token | "A word" | A unit in the model's vocabulary - could be a character, subword, word, or multi-word chunk |
| BPE | "Some compression thing" | Byte Pair Encoding - iteratively merge the most frequent adjacent pair of tokens |
| WordPiece | "BERT's tokenizer" | Like BPE but merges maximize training data likelihood instead of raw frequency |
| SentencePiece | "A tokenizer library" | A language-agnostic tokenizer that operates on raw Unicode, supporting BPE and Unigram algorithms |
| Vocabulary size | "How many words it knows" | The total number of unique tokens the model can represent - typically 30K to 100K |
| Fertility | "Not a tokenizer term" | Average number of tokens per word - measures tokenizer efficiency across languages |
| Byte-level BPE | "GPT's tokenizer" | BPE operating on raw bytes (0-255) instead of Unicode characters - guarantees no unknown tokens |
| Merge table | "The tokenizer file" | Ordered list of pair merges learned during training - this IS the tokenizer |
| Pre-tokenization | "Splitting on spaces" | Rules applied before subword tokenization: whitespace splitting, digit separation, punctuation handling |
| tiktoken | "OpenAI's tokenizer" | OpenAI's fast BPE implementation used by GPT-3.5/4, with ~100K vocabulary |

## Further Reading

- [Sennrich et al., 2016 - Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - the paper that introduced BPE for NLP
- [Kudo & Richardson, 2018 - SentencePiece](https://arxiv.org/abs/1808.06226) - language-agnostic subword tokenization
- [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers) - production-grade tokenizer training
- [Andrej Karpathy's minbpe](https://github.com/karpathy/minbpe) - minimal BPE implementation for education
- [tiktoken source code](https://github.com/openai/tiktoken) - OpenAI's Rust+Python BPE tokenizer
