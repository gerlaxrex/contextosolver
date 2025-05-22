from typing import Optional, List

from pydantic import BaseModel


class ContextoConfig(BaseModel):
    initial_words: Optional[List[str]] = None
    top_n: int = 100
    max_guesses: int = 200
    warmup_words: int = 10
    top_k_words: int = 5
    top_neg_words: int = 100
    negative_threshold: int = 2000
    negative_penalty_weight: float = 0.8

    class Config:
        extras = "ignore"
