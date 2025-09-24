# Specific guidelines and checks at all times

## General checks
- Python 3.12 type hinting without typing package reliance, if anything remove them never include them in any of the code.
- Avoid any docstring in the code unless specified or comments.
- Always use pydantic model whenever structured output is required, use it across the entire repo and do not use any dataclasses.
- Use Enum type when needed, avoid using strings and constants.
- Package imports on top of python modules, never within any function or classes.
- Avoid creating new python modules for duplicated functionalities.
- Are the markdown files for design docs fully updated with implementations? If not, update them at all times.
- Whenever you are logging or showing output, dont use [:..] to only take the first number of tokens, show the entire content. Never use any truncation in anywhere in the code.
- Consolidate common patterns into reusable utility functions to avoid code duplication.
- When working with external APIs, implement dynamic error handling rather than hardcoding model-specific behavior.
- Group related functionality in logical modules (e.g., agent utilities in agents_research.py, core services in services.py).

## OpenAI Agents SDK Requirements
- MUST use `from agents import Agent, Runner, WebSearchTool` for all agent implementations.
- Agent creation: `Agent(name="AgentName", model="gpt-4o", instructions="...", tools=[...])`
- Execution patterns: `Runner.run_sync(agent, prompt)` or `Runner.run_streamed(agent, prompt)`  
- For streaming: `async for event in stream.stream_events()` to handle text deltas
- MUST use Pydantic models for all data structures - no dataclasses, no plain dicts.
- MUST use proper Python 3.12+ type annotations on all functions and variables.
- Follow patterns from parallel_agents implementation for consistency.
- Don't use mocks, use the API keys and return full structured outputs, don't take the first "n" tokens.

## Functional Programming
- Whenever possible use functions as opposed to create OOP, much cleaner to read and we probably don't need all that inheritance and design patterns, unless otherwise specified for specific use cases.
- Functions should be correctly decoupled when too complex within common python modules.

## README Documentation Guidelines
- Keep mermaid diagrams minimal and essential - maximum 3 diagrams total
- Required diagrams: 1) System Architecture, 2) Agent Flow/Dependencies, 3) Sequence Diagram
- Remove redundant flowcharts that duplicate information shown in other diagrams  
- Focus on showing how components interact, not internal implementation details
- Use clear, descriptive titles that explain the diagram's purpose
- Maintain consistent styling and color schemes across all diagrams