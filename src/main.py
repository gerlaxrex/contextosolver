import json

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
import os
import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf, DictConfig

from contextosolver import ROOT_PATH
from contextosolver.solver.contexto_client import ContextoClient
from contextosolver.solver.contexto_solver import ContextoConfig, ContextoSolver


def run(game_id: int, config: ContextoConfig):
    embedding_model = TextEmbedding()
    qdrant_client = QdrantClient(
        os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    contexto_client = ContextoClient(game_id=game_id)

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


@hydra.main(config_path=f"{ROOT_PATH}/config", config_name="config", version_base=None)
def main(config: DictConfig):
    load_dotenv()
    game_id = config["game_id"]
    logger.info(f"Starting for game #{game_id}!")
    config = ContextoConfig(**OmegaConf.to_container(config, resolve=True))
    logger.info(f"Running with Configs:\n{json.dumps(config.model_dump(), indent=2)}")
    run(game_id, config)


if __name__ == "__main__":
    from pyfiglet import Figlet

    f = Figlet(font="slant")
    print(f.renderText("Contexto Solver"))
    main()
