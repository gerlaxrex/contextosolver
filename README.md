# Contexto Solver

This repository contains a solver for the game contexto.me. It uses word embeddings, Qdrant as a vector search engine, and a bit of strategy to guess the secret word efficiently.

## Overview

The idea is to treat this as a sort of search problem: we try a word, get a ranking back (how close we are to the target), and use that feedback to guide our next guess. The ranking acts like a reward signal, and we use the top-performing guesses to search semantically similar words in embedding space.

## Preprocessing

I started with a large English word list from another GitHub repo (not linked here), and used SpaCy to lemmatize the words—contexto ranks based on lemmas, not inflected forms. This reduced the word count by about 46%. After that, embeddings were generated using fastembed, and indexed in a private Qdrant collection (this can also be used locally by specifying the local path to the Qdrant DB).

The scripts for lemmatization and data indexing can be found in the preprocessing module.

## Solver Strategy

The solver relies on:

* An embedding model (via fastembed)
* A Qdrant client to perform nearest-neighbor queries
* A lightweight client that calls the contexto.me game API to retrieve rankings

Main strategy:

1. Start with some fixed "warmup" words (found online as good general-purpose openers)
2. Use the top-k best guesses to compute a weighted average embedding that works as a centroid
3. Query Qdrant with that embedding to get similar candidates
4. Penalize words that are close to previously bad guesses (blacklist region)
5. Repeat until the target is found or max guesses is reached

The solver also handles some edge cases, like when no useful candidates are found or a word returns an invalid response.

## Running the Solver

Make sure you have:

* A running Qdrant instance (local, dockerized or hosted) with the preloaded collection

Then just run:

```bash
python src/contextosolver/main.py
```

This will attempt to solve a specific game (you can change the game ID in the script).
