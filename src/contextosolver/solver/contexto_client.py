from typing import Optional

import httpx
from pydantic import BaseModel


class ContextoResponse(BaseModel):
    word: str
    lemma: str
    distance: int


class ContextoClient:
    def __init__(self, game_id: int):
        self.game_id = game_id
        self.__base_url = f"https://api.contexto.me/machado/en/game/{self.game_id}"
        self._client = httpx.Client()

    def get_response(self, word: str) -> Optional[ContextoResponse]:
        try:
            res = self._client.get(
                url=f"{self.__base_url}/{word.strip()}",
            )
            return ContextoResponse(**res.json())
        except Exception:
            return None


if __name__ == "__main__":
    contexto_client = ContextoClient(game_id=973)
    res = contexto_client.get_response("event")
    print(res.distance)
