from pydantic import BaseModel

class MatchInput(BaseModel):
    home_form: float = 1.5
    away_form: float = 1.2
    home_fixture_density: int = 2
    away_fixture_density: int = 2
    elo_delta: float = 0.0
    home_advantage: int = 1
    B365H: float = 2.0
    B365D: float = 3.4
    B365A: float = 3.8

class PlayerRatingOutput(BaseModel):
    player_id: int
    performance_score: float

class TopPerformersInput(BaseModel):
    metric: str = "finishing"
    limit: int = 10