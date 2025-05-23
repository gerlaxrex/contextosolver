import json
from typing import List

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from loguru import logger
import pathlib as pl


def load_words_on_db(
    qdrant_client: QdrantClient,
    embedding_model: TextEmbedding,
    collection_name: str,
    word_list_path: str = None,
    word_list: List[str] = None,
    recreate: bool = False,
):
    # for checkpointing
    starting_batch = 0
    checkpoint = {}
    filename = pl.Path(word_list_path).name

    if (DATA_PATH / "checkpoint.json").exists():
        with open(DATA_PATH / "checkpoint.json", "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
            starting_batch = checkpoint.get(filename, 0)

    # Get words from file or direct list
    if word_list_path and pl.Path(word_list_path).exists():
        with open(word_list_path, "r", encoding="utf-8") as f:
            # this assumes that the file is a json file that contains a list
            words = json.load(f)
    elif word_list:
        words = word_list

    logger.info(f"Loaded {len(words)} words")

    embedding_dim = len(list(embedding_model.embed("test"))[0])

    if (
        starting_batch == 0
        and recreate
        and qdrant_client.collection_exists(collection_name=collection_name)
    ):
        qdrant_client.delete_collection(collection_name=collection_name)

    if not qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            timeout=100,
            # hnsw_config=models.HnswConfig(
            #     m=60,
            #     ef_construct=500,
            #     full_scan_threshold=500,
            # ),
        )

    # Process in batches to avoid memory issues
    batch_size = 100
    total_batches = (len(words) + batch_size - 1) // batch_size

    for batch_idx in range(starting_batch, total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(words))
        batch_words = words[start_idx:end_idx]

        embeddings = embedding_model.embed(batch_words)

        points = [
            models.PointStruct(
                id=i + start_idx, vector=embedding.tolist(), payload={"word": word}
            )
            for i, (word, embedding) in enumerate(zip(batch_words, embeddings))
        ]

        qdrant_client.upsert(collection_name=collection_name, points=points)

        logger.info(
            f"Processed batch {batch_idx + 1}/{total_batches} ({len(batch_words)} words)"
        )
        checkpoint[filename] = batch_idx

        with open(DATA_PATH / "checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(checkpoint, f)

    checkpoint[filename] = 0
    with open(DATA_PATH / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)

    logger.info(f"Created vector database with {len(words)} words")


if __name__ == "__main__":
    from src.contextosolver import DATA_PATH
    import os
    from dotenv import load_dotenv

    load_dotenv()

    embedding_model = TextEmbedding(model_name="BAAI/bge-base-en")
    qdrant_client = QdrantClient(
        os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), https=True
    )

    load_words_on_db(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        collection_name="contexto_words",
        word_list_path=DATA_PATH / "unique_lemmas.json",
        recreate=True,
    )
