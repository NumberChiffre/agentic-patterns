from __future__ import annotations

from agents import Agent, WebSearchTool

from ..core.types import JudgeVerdict


def make_candidate(
    name: str,
    model: str,
    instructions: str,
    tools: list[object] | None = None,
    enable_web_search: bool = True,
) -> Agent:
    return Agent(
        name=name,
        model=model,
        instructions=instructions,
        tools=tools
        if tools is not None
        else ([WebSearchTool()] if enable_web_search else []),
    )


def make_full(agent: Agent, instructions: str) -> Agent:
    return Agent(
        name=f"{agent.name} (full)",
        model=agent.model,
        instructions=instructions,
        tools=agent.tools,
    )


def make_judge(model: str, instructions: str) -> Agent:
    return Agent(
        name="Judge",
        model=model,
        instructions=instructions,
        output_type=JudgeVerdict,
    )
