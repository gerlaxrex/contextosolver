import argparse

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
import os
from contextosolver.solver.contexto_client import ContextoClient
from contextosolver.solver.contexto_solver import ContextoConfig, ContextoSolver


def main(game_id: int):
    embedding_model = TextEmbedding()
    qdrant_client = QdrantClient(
        os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    contexto_client = ContextoClient(game_id=game_id)

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
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run the game with a specific game ID."
    )
    parser.add_argument("game_id", type=str, help="The ID of the game to run")
    args = parser.parse_args()
    main(args.game_id)
