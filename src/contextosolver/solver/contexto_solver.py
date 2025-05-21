import numpy as np
from pydantic import BaseModel
from sklearn.cluster import KMeans
import heapq
import json
import random
import time
from typing import Optional, List, Tuple, Dict

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from loguru import logger

from src.contextosolver import DATA_PATH
from src.contextosolver.solver.contexto_client import ContextoClient

MAX_SCORE = 500_000


class ContextoConfig(BaseModel):
    initial_words: Optional[List[str]] = None
    top_n: int = 10
    max_guesses: int = 100
    warmup_words: int = 10
    top_k_words: int = 5
    max_neg_words: int = 50
    top_neg_words: int = 100
    negative_threshold: int = 2000
    negative_penalty_weight: float = 0.8


class ContextoSolver:
    def __init__(
        self,
        embedding_model: TextEmbedding,
        qdrant_client: QdrantClient,
        contexto_client: ContextoClient,
        collection_name: str = "contexto_words",
        initial_words: Optional[List[str]] = None,
        *,
        config: Optional[ContextoConfig] = None,
    ):
        self.embedding_model = embedding_model
        self.embedding_dim = len(list(self.embedding_model.embed("test"))[0])
        self.qdrant_client = qdrant_client
        self.contexto_client = contexto_client
        self.config = config

        with open(DATA_PATH / "unique_lemmas.json", "r", encoding="utf-8") as f:
            self.lemmas = json.load(f)

        self.collection_name = collection_name

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
        if self.initial_words and len(self.guessed_words) < self.config.warmup_words:
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

    def _clustered_negative_penalty(self, candidate_scores: Dict[str, float]) -> None:
        # Perform the clustering over all the blacklisted and not only on a small subset
        blacklisted_items = list(self.blacklisted_words.items())
        blacklist_words = [word for word, _ in blacklisted_items]
        blacklist_rankings = [rank for _, rank in blacklisted_items]
        blacklist_embeddings = list(self.embedding_model.embed(blacklist_words))
        np_blacklist = np.array(blacklist_embeddings)

        if len(blacklist_words) < 10:
            num_clusters = 2
        elif len(blacklist_words) < 20:
            num_clusters = 3
        else:
            num_clusters = min(5, len(blacklist_words) // 5)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(np_blacklist)

        for cluster_idx in range(num_clusters):
            cluster_member_indices = [
                i for i, c in enumerate(clusters) if c == cluster_idx
            ]

            # empty clusters
            if not cluster_member_indices:
                continue

            cluster_words = [blacklist_words[i] for i in cluster_member_indices]
            cluster_centroid = kmeans.cluster_centers_[cluster_idx]
            # how many words in the cluster
            cluster_influence = len(cluster_words) / len(blacklist_words)
            avg_cluster_ranking = sum(
                [blacklist_rankings[i] for i in cluster_member_indices]
            ) / len(cluster_words)
            # also score for the average rankings!
            if max(blacklist_rankings) == min(blacklist_rankings):
                cluster_ranking_penalty = 1.0
            else:
                cluster_ranking_penalty = (
                    avg_cluster_ranking - min(blacklist_rankings)
                ) / (max(blacklist_rankings) - min(blacklist_rankings))

            try:
                negative_results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=cluster_centroid.tolist(),
                    limit=self.config.top_neg_words,
                )
                for result in negative_results.points:
                    candidate = result.payload["word"]

                    if candidate in candidate_scores:
                        similarity = result.score
                        penalty = (
                            self.config.negative_penalty_weight
                            * similarity
                            * cluster_influence
                            * cluster_ranking_penalty
                        )
                        max_penalty = 0.75 * candidate_scores[candidate]
                        candidate_scores[candidate] -= min(penalty, max_penalty)

            except Exception as e:
                logger.error(f"Error querying for cluster {cluster_idx}: {e}")

    def _enhanced_select_best_candidate(self) -> str:
        # if we did not have any guessed words then go random
        if not self.guessed_words:
            return random.choice(self.initial_words)

        # take the top_k_words for the ones currently explored
        best_guesses = heapq.nsmallest(
            min(self.config.top_k_words, len(self.guessed_words)),
            self.guessed_words.items(),
            key=lambda x: x[1],
        )

        # take the negative matches
        # worst_guesses = heapq.nsmallest(
        #     min(self.config.max_neg_words, len(self.blacklisted_words)),
        #     self.blacklisted_words.items(),
        #     key=lambda x: x[1],
        # )

        # inverse ranking weighting average
        centroid_embedding = self._weighted_embeddings(best_guesses)
        candidate_scores = {}

        # find nbs for the avg vector
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=centroid_embedding,
            limit=self.config.top_n,
            search_params=models.SearchParams(hnsw_ef=600),
        )

        for result in search_results.points:
            candidate = result.payload["word"]
            if candidate not in self.words_to_exclude:
                candidate_scores[candidate] = result.score

        # now take the average for the clusters obtained in the blacklisted words
        # blacklist_embedding = self._weighted_embeddings(sorted(worst_guesses, key=lambda x: x[1], reverse=True))
        #
        # negative_results = self.qdrant_client.query_points(
        #     collection_name=self.collection_name,
        #     query=blacklist_embedding,
        #     limit=self.config.top_neg_words,
        # )
        #
        # for negative_result in negative_results.points:
        #     candidate = negative_result.payload["word"]
        #     if candidate in candidate_scores:
        #         candidate_scores[candidate] -= negative_result.score
        self._clustered_negative_push(candidate_scores=candidate_scores)

        if not candidate_scores:
            return self._get_random_word()

        best_candidate = max(candidate_scores.items(), key=lambda x: x[1])[0]
        return best_candidate

    def update_with_feedback(self, word: str, lemma: str, ranking: int):
        self.guessed_words[lemma] = ranking
        self.words_to_exclude.add(word)
        self.words_to_exclude.add(lemma)

        if ranking > self.config.negative_threshold:
            # it is needed for excluding sub-regions of embeddings
            self.blacklisted_words[lemma] = ranking

    def solve(self):
        guesses = []
        start_time = time.time()

        for i in range(self.config.max_guesses):
            next_word = self.get_next_guess()
            ranking = self.contexto_client.get_response(next_word)

            if not ranking:
                # in case there is an error or word does not exists
                self.update_with_feedback(next_word, next_word, MAX_SCORE)
                guesses.append((next_word, MAX_SCORE))
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
    contexto_client = ContextoClient(game_id=975)

    config = ContextoConfig(
        max_guesses=200,
        warmup_words=12,
        top_n=500,
        top_k_words=3,
    )

    # create the solver
    contexto_solver = ContextoSolver(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        contexto_client=contexto_client,
        collection_name="contexto_words",
        config=config,
    )

    result = contexto_solver.solve()
