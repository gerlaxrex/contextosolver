from typing import Optional, List

from pydantic import BaseModel, Field


class ContextoConfig(BaseModel):
    initial_words: Optional[List[str]] = Field(
        default=None, description="list of initial words for the warmup"
    )
    top_n: int = Field(
        default=100,
        description="The number of similar words to be extracted for the positive centroid",
    )
    max_guesses: int = Field(
        default=200, description="The maximum number of guesses to be taken"
    )
    warmup_words: int = Field(
        default=10, description="The number of warmup words to try at the beginning"
    )
    top_k_words: int = Field(
        default=5,
        description="The top ranked words found in order to build the centroid embedding",
    )
    top_neg_words: int = Field(
        default=100,
        description="The number of similar words to be extracted for each negative centroid cluster",
    )
    negative_threshold: int = Field(
        default=2000, description="The rank that thresholds the blacklisted words"
    )
    negative_penalty_weight: float = Field(
        default=0.8, description="The penalty weight for negative words"
    )

    class Config:
        extras = "ignore"
