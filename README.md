# steppelm

SteppeLM is an educational project for building a small Turkmen language
model from openly available Turkmen text data.

## Project structure

- `README.md` — project documentation
- `requirements.txt` — Python dependencies
- `data/tm-data/` — Turkmen source dataset provided as a Git submodule
- `data/processed/` — generated training datasets
- `src/` — dataset and model source code
- `tests/` — automated tests
- `checkpoints/` — model checkpoints used by later tasks

## Source dataset

The initial dataset is provided by:

https://github.com/turkmenos/tm-data

The SteppeLM repository consumes the `stories/` JSON files from `tm-data`.

The story dataset uses page-level JSON documents with a `pages` array.
Each page contains fields such as `page_number` and `text`.

The source dataset remains separate from the generated training dataset.

## Initialize the dataset

After cloning the repository:

```bash
git submodule update --init --recursive