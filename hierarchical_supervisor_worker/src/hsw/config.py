import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class HswConfig(BaseModel):
    openai_api_key: str
    redis_url: str
    max_depth: int
    max_parallel: int
    max_tokens: int
    max_seconds: int

def load_config() -> HswConfig:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY must be provided in .env file")
    
    return HswConfig(
        openai_api_key=openai_api_key,
        redis_url=os.getenv("REDIS_URL") or "redis://localhost:6379/0",
        max_depth=int(os.getenv("HSW_MAX_DEPTH") or "2"),
        max_parallel=int(os.getenv("HSW_MAX_PARALLEL") or "3"),
        max_tokens=int(os.getenv("HSW_MAX_TOKENS") or "8000"),
        max_seconds=int(os.getenv("HSW_MAX_SECONDS") or "20"),
    )
