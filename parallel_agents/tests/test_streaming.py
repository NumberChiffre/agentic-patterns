from __future__ import annotations

import pytest

from src.streaming import agentsdk_text_stream, stream_response


@pytest.mark.asyncio
async def test_agentsdk_text_stream_yields_text_chunks() -> None:
    from agents import Agent

    agent = Agent(name="A", model="gpt-FAKE", instructions="hi")
    chunks = []
    async for c in agentsdk_text_stream(agent, "prompt"):
        chunks.append(c)
    assert chunks, "expected at least one chunk"


@pytest.mark.asyncio
async def test_stream_response_counts_tokens_and_can_capture_text() -> None:
    from agents import Agent

    agent = Agent(name="A", model="gpt-FAKE", instructions="hi")
    tokens, text = await stream_response(agent, "prompt", stop_after_tokens=5, capture_text=True, log_every_tokens=1)
    assert isinstance(tokens, int) and tokens >= 1
    assert isinstance(text, str)


