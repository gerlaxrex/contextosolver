import heapq
import json
import random
import time
from typing import Optional, List, Tuple

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from loguru import logger

from src.contextosolver import DATA_PATH
from src.contextosolver.solver.contexto_client import ContextoClient


class ContextoSolver:
    def __init__(
        self,
        embedding_model: TextEmbedding,
        qdrant_client: QdrantClient,
        contexto_client: ContextoClient,
        collection_name: str = "contexto_words",
        initial_words: Optional[List[str]] = None,
        top_n: int = 10,
        max_guesses: int = 100,
        warmup_words: int = 10,
        top_k_words: int = 5,
    ):
        self.embedding_model = embedding_model
        self.embedding_dim = len(list(self.embedding_model.embed("test"))[0])
        self.qdrant_client = qdrant_client
        self.contexto_client = contexto_client
        self.max_guesses = max_guesses
        self.warmup_words = warmup_words
        self.top_k_words = top_k_words

        with open(DATA_PATH / "unique_lemmas.json", "r", encoding="utf-8") as f:
            self.lemmas = json.load(f)

        self.collection_name = collection_name
        self.top_n = top_n

        # https://wordfinder.yourdictionary.com/blog/contexto-tips-strategies-best-starting-words/ eheh
        self.initial_words = initial_words or [
            "person",
            "place",
            "thing",
            "idea",
            "food",
            "occupation",
            "animal",
            "event",
            "city",
            "nature",
            "home",
            "work",
            "technology",
        ]

        self.guessed_words = {}
        self.words_to_exclude = set()
        # this lists represent:
        # - the words that have very low scores
        # - exclude words given eventual sub-zones of the blacklisted words
        self.blacklisted_words = {}

    def _get_random_word(self) -> str:
        random_word = random.choice(self.lemmas)
        while random_word in self.guessed_words:
            random_word = random.choice(self.lemmas)
        return random_word

    def get_next_guess(self) -> str:
        # first use randomly some initial word
        if self.initial_words and len(self.guessed_words) < self.warmup_words:
            return self.initial_words[len(self.guessed_words)]

        # semantic search on qdrant
        return self._enhanced_select_best_candidate()

    def _weighted_embeddings(self, ranked_words: List[Tuple[str, int]]) -> List[float]:
        words = [word for word, _ in ranked_words]
        vectors = list(self.embedding_model.embed(words))
        ranks = [rank for _, rank in ranked_words]
        # weights on the rank
        if max(ranks) == min(ranks):
            weights = [1.0] * len(ranks)
        else:
            weights = [
                (max(ranks) - rank) / (max(ranks) - min(ranks)) for rank in ranks
            ]
        weighted_sum = sum(w * v for w, v in zip(weights, vectors))
        centroid_embedding = weighted_sum / sum(weights)
        return centroid_embedding

    def _enhanced_select_best_candidate(self) -> str:
        # if we did not have any guessed words then go random
        if not self.guessed_words:
            return random.choice(self.initial_words)

        # take the top_k_words for the ones currently explored
        best_guesses = heapq.nsmallest(
            min(self.top_k_words, len(self.guessed_words)),
            self.guessed_words.items(),
            key=lambda x: x[1],
        )

        # take the negative matches
        worst_guesses = heapq.nsmallest(
            min(50, len(self.blacklisted_words)),
            self.blacklisted_words.items(),
            key=lambda x: x[1],
        )

        # inverse ranking weighting average
        centroid_embedding = self._weighted_embeddings(best_guesses)
        candidate_scores = {}

        # find nbs for the avg vector
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=centroid_embedding,
            limit=self.top_n,
            search_params=models.SearchParams(hnsw_ef=600),
        )

        for result in search_results.points:
            candidate = result.payload["word"]
            if candidate not in self.words_to_exclude:
                candidate_scores[candidate] = result.score

        # now take the average for the clusters obtained in the blacklisted words
        blacklist_embedding = self._weighted_embeddings(worst_guesses)

        negative_results = self.qdrant_client.query_points(
            collection_name="contexto_words",
            query=blacklist_embedding,
            limit=100,
        )

        for negative_result in negative_results.points:
            candidate = negative_result.payload["word"]
            if candidate in candidate_scores:
                candidate_scores[candidate] -= negative_result.score

        if not candidate_scores:
            return self._get_random_word()

        best_candidate = max(candidate_scores.items(), key=lambda x: x[1])[0]
        return best_candidate

    def update_with_feedback(self, word: str, lemma: str, ranking: int):
        self.guessed_words[lemma] = ranking
        self.words_to_exclude.add(word)
        self.words_to_exclude.add(lemma)

        if ranking > 2000:
            # it is needed for excluding sub-regions of embeddings
            self.blacklisted_words[lemma] = ranking

    def solve(self):
        guesses = []
        start_time = time.time()

        for i in range(self.max_guesses):
            next_word = self.get_next_guess()
            ranking = self.contexto_client.get_response(next_word)

            if not ranking:
                # in case there is an error or word does not exists
                self.update_with_feedback(next_word, next_word, 500_000)
                guesses.append((next_word, 500_000))
                logger.warning(f"word {next_word} does not exist!")
                continue

            self.update_with_feedback(ranking.word, ranking.lemma, ranking.distance)
            guesses.append((ranking.word, ranking.distance))
            logger.info(f"Guess {i + 1}: {ranking}")

            if ranking.distance == 0:
                logger.info(
                    f"\nFound the target word '{next_word}' in {i + 1} guesses!"
                )
                logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
                break

        return guesses


if __name__ == "__main__":
    import os

    # dependencies
    embedding_model = TextEmbedding()
    qdrant_client = QdrantClient(
        os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    contexto_client = ContextoClient(game_id=974)

    # create the solver
    contexto_solver = ContextoSolver(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        contexto_client=contexto_client,
        collection_name="contexto_words",
        max_guesses=500,
        warmup_words=12,
        top_n=1000,
        top_k_words=4,
    )

    result = contexto_solver.solve()
