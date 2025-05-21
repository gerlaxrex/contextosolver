# Contexto Solver

This repository contains a solver for the game [contexto.me](https://contexto.me). It uses word embeddings, [Qdrant](https://qdrant.tech) as a vector search engine, and a strategic guessing process to identify the secret word.

## Overview

Each guess in Contexto returns a ranking that indicates how semantically close the word (more specifically, its __lemma__) is to the hidden target. This solver treats the problem as a search over a semantic space, using feedback from each guess to guide the next.

> [!NOTE]
> There’s a light parallel with **reinforcement learning**: guesses act as actions, the rankings resemble reward signals, and the embedding space serves as the environment. While there’s no learning involved, the feedback loop and the warmup stage imitates the exploration–exploitation trade-off.

## Preprocessing

I began with a large English word list, then used [SpaCy](https://spacy.io) to lemmatize the words — since Contexto evaluates based on lemmas, not inflected forms. This reduced the vocabulary size significantly (~46%).

Embeddings were then generated using [fastembed](https://github.com/qdrant/fastembed) and stored in a Qdrant collection for efficient vector similarity search.

Scripts for lemmatization and indexing are located in the `preprocessing` module.

## Solver Strategy

The solver is composed of:

* A `fastembed`-based embedding model
* A Qdrant client to perform similarity search
* A Contexto client that fetches ranking feedback from the game using the contexto.me apis!

### Core Steps:

1. **Warm-Up Phase**
   The solver starts with a list of general-purpose starter words (e.g., *person*, *place*, *idea*, *animal*). These are broad enough to provide useful feedback early on.

2. **Centroid-Based Search**
   From the top-performing previous guesses, it computes a weighted centroid embedding — where better-ranked guesses have higher weight. This acts as the next semantic search direction. A min-max scaling is used for the weights for the ranking.

3. **Nearest Neighbor Retrieval**
   The centroid is used to query Qdrant for the most semantically similar words.

4. **Negative Feedback: penalizing bad words**:
   * Poor guesses are clustered in embedding space using KMeans.
   * Each cluster is scored by how poor its members are.
   * Candidates close to these "bad regions" receive a penalty.
   * The penalty is scaled by similarity to the cluster, cluster size, and average rank.

   This helps the solver avoid repeatedly guessing within unproductive subregions of the vector space.

5. **Fallback and Randomization**
   If no strong candidates are found, or all high-similarity candidates are blacklisted, a random unseen word is selected to keep the exploration moving.

## Running the Solver

As a first step install all the dependencies (I used uv as a package manager):

```bash
uv venv
uv sync
```

Ensure that:

* A Qdrant instance (local or remote) is running and has the preloaded collection
* Environment variables for `QDRANT_URL` and `QDRANT_API_KEY` are set (if needed) in a `.env` file

Then in the `src` folder run:

```bash
uv run main.py <game-id>
```

This will launch the solver on the specified game.

## Tips

* The number and quality of initial warmup words affect early convergence.
* The negative penalty mechanism becomes most effective in the mid-to-late game when enough poor guesses have accumulated.
* Adjust `ContextoConfig` to tune search aggressiveness (e.g., `top_k_words`, `top_n`, and penalty weights).
