## Train the first Turkmen tokenizer

SteppeLM uses a Byte-Pair Encoding (BPE) tokenizer for the initial tokenizer.

BPE learns a vocabulary by starting from small symbols and repeatedly merging
frequent token pairs. This produces subword tokens that can represent common
Turkmen words while still handling previously unseen words.

The tokenizer defines these special tokens:

- `<pad>` — padding
- `<unk>` — unknown token
- `<bos>` — beginning of sequence
- `<eos>` — end of sequence
- `<mask>` — masked token reserved for future use

Install dependencies:

```bash
python -m pip install -r requirements.txt