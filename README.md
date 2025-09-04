<div align="center">

  <h2>Agentic Patterns ✨</h2>
  <p>A friendly collection of reusable agentic patterns and workflows that make LLMs feel faster, smarter, and more reliable.</p>

  <p>
    <b>Goals:</b> ⚡ Speed · 🛡️ Reliability · 💸 Cost · 👀 Observability · 🧩 Composability
  </p>

</div>

---

### Overview

This repo hosts practical agentic patterns you can copy, adapt, and compose into your own systems — built from scratch with minimal dependencies. Each pattern focuses on real product concerns and avoids heavy frameworks.

- ⚡ Lower latency with parallelism and early stopping
- 🛡️ Higher reliability with structured fallbacks and judging
- 💸 Lower cost by spending where it matters (previews vs full answers)
- 👀 Better observability with logs/metrics/hooks
- 🧩 Composable pieces you can mix-and-match

### Philosophy (clear and simple)

- 🏗️ **Barebones-first**: Implementations from scratch. No LangChain or similar orchestration frameworks.
- 🧰 **SDK-first**: Prefer official SDKs and thin clients (e.g., OpenAI Agents SDK), or direct HTTP calls when simpler.
- 🪶 **Minimal deps**: Small, readable modules over complex abstractions.
- 🔬 **Transparent control**: Explicit routing, retries, and evaluation; no hidden magic.
- 🔌 **Portable**: Patterns you can lift into any stack or infra.

### Non-goals (what this repo is not)

- 🚫 A framework or all-in-one runtime
- 🚫 A plugin marketplace or prompt-in-a-box solution
- 🚫 Vendor lock-in; patterns should generalize across providers

### Who is this for?

- Engineers who want to understand and control agent behavior end-to-end
- Teams who prefer direct SDKs over large orchestration layers
- Builders who value clarity, reliability, and production-minded patterns

### Contributing

Contributions are welcome! Please:

- Keep modules self‑contained with a clear `README.md` and minimal dependencies
- Prefer small, composable abstractions over frameworks
- Include a quickstart and a realistic example per module

If you’re unsure where to start, open a discussion or a small draft PR — we’re friendly 🙂

### License

MIT License — see `LICENSE`.

