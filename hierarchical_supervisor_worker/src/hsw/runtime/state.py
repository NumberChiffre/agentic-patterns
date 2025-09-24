import json
from pydantic import BaseModel, ConfigDict
import redis

class StateManager(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    redis_client: redis.Redis

    @classmethod
    def from_url(cls, redis_url: str) -> "StateManager":
        client = redis.from_url(redis_url, decode_responses=True)
        return cls(redis_client=client)

    def store_json(self, key: str, value: str | int | list | dict, ttl_seconds: int = 3600) -> None:
        self.redis_client.setex(key, ttl_seconds, json.dumps(value, default=str))

    def get_json(self, key: str) -> str | int | list | dict | None:
        raw = self.redis_client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
