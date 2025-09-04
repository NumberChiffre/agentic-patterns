from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator



# Install a minimal fake 'agents' package into sys.modules BEFORE tests import src.*
import sys
import types


@dataclass
class FakeRunConfig:
    tracing_disabled: bool = True


@dataclass
class FakeTool:
    name: str = "fake_tool"


@dataclass
class FakeAgent:
    name: str
    model: str
    instructions: str
    tools: list[Any] | None = None
    output_type: Any | None = None

    async def get_all_tools(self, context_wrapper: Any) -> list[Any]:
        return list(self.tools or [])


class _Ev:
    def __init__(self, type: str, text: str | None = None):
        self.type = type
        self.text = text
        self.data = None
        self.raw_item = None


class FakeStream:
    def __init__(self, events: list[_Ev]):
        self._events = events

    async def stream_events(self) -> AsyncIterator[_Ev]:
        for e in self._events:
            await asyncio.sleep(0)
            yield e


class FakeRunner:
    @staticmethod
    def run_streamed(agent: FakeAgent, prompt: str, run_config: FakeRunConfig | None = None) -> FakeStream:
        # For previews/judge: emit one JSON chunk; for general streaming: emit a few words
        if "winner_index" in prompt or prompt.strip().startswith("{"):
            events = [_Ev("text_delta_event", "{\"winner_index\": 0, \"scores\": [{\"index\": 0, \"relevance\": 1, \"coverage\": 1, \"faithfulness\": 1, \"overall\": 1}]}")]
        else:
            events = [
                _Ev("text_delta_event", "hello "),
                _Ev("text_delta_event", "world "),
                _Ev("text_delta_event", "from fake"),
            ]
        return FakeStream(events)


fake_mod = types.ModuleType("agents")
fake_mod.Agent = FakeAgent
fake_mod.RunConfig = FakeRunConfig
fake_mod.Runner = FakeRunner
fake_mod.WebSearchTool = FakeTool
sys.modules.setdefault("agents", fake_mod)



