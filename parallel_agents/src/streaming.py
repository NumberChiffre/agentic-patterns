from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from agents import Agent, RunConfig, Runner
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .types import (
    AnnotationType,
    DELTA_DATA_TYPE_SUBSTRINGS,
    RawDataCategory,
    StreamEventType,
)


logger = logging.getLogger(__name__)


def _count_tokens(s: str) -> int:
    return 0 if not s else len(s.split())


async def agentsdk_text_stream(agent: Agent, prompt: str) -> AsyncIterator[str]:
    stream = Runner.run_streamed(agent, prompt, run_config=RunConfig(tracing_disabled=True))
    async for ev in stream.stream_events():
        ev_type = str(getattr(ev, "type", ""))

        if ev_type == StreamEventType.TEXT_DELTA_EVENT:
            text = getattr(ev, "text", None)
            if isinstance(text, str) and text:
                yield text

        elif ev_type == StreamEventType.RAW_RESPONSE_EVENT:
            data = getattr(ev, "data", None)
            data_type_str = str(getattr(data, "type", ""))
            if any(sub in data_type_str for sub in DELTA_DATA_TYPE_SUBSTRINGS):
                delta = getattr(data, "delta", None)
                text = (
                    delta
                    or getattr(data, "text", None)
                    or getattr(data, "content", None)
                )
                if isinstance(text, str) and text:
                    yield text

            elif RawDataCategory.WEB_SEARCH_CALL in data_type_str:
                try:
                    if RawDataCategory.COMPLETED in data_type_str:
                        if hasattr(data, "item") and hasattr(data.item, "web_search_result"):
                            search_result = data.item.web_search_result
                            if hasattr(search_result, "results"):
                                for r in search_result.results:
                                    url = getattr(r, "url", "")
                                    title = getattr(r, "title", "")
                                    if url and title:
                                        logger.info(f"🔗 Citation found: {title} - {url}")
                        if hasattr(data, "results"):
                            for r in data.results:
                                url = getattr(r, "url", "")
                                title = getattr(r, "title", "")
                                if url and title:
                                    logger.info(f"🔗 Citation found: {title} - {url}")
                except Exception as citation_err:  # noqa: BLE001
                    logger.debug(f"Citation parse skipped: {citation_err}")

            elif RawDataCategory.ANNOTATION_ADDED in data_type_str:
                try:
                    ann = getattr(data, "annotation", None)
                    if isinstance(ann, dict) and ann.get("type") == AnnotationType.URL_CITATION:
                        title = str(ann.get("title", ""))
                        url = str(ann.get("url", ""))
                        if url and title:
                            logger.info(f"🔗 Citation found (annotation): {title} - {url}")
                    elif hasattr(ann, "type") and getattr(ann, "type", "") == AnnotationType.URL_CITATION:
                        title = str(getattr(ann, "title", ""))
                        url = str(getattr(ann, "url", ""))
                        if url and title:
                            logger.info(f"🔗 Citation found (annotation): {title} - {url}")
                except Exception as ann_err:  # noqa: BLE001
                    logger.debug(f"Annotation parse skipped: {ann_err}")

        elif ev_type == StreamEventType.MESSAGE_OUTPUT_ITEM:
            parts = getattr(getattr(ev, "raw_item", None), "content", []) or []
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t:
                    yield t


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential_jitter(initial=0.2, max=3.0),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def stream_response(
    agent: Agent,
    prompt: str,
    *,
    stop_after_tokens: int | None = None,
    log_every_tokens: int = 50,
    capture_text: bool = False,
) -> tuple[int, str | None]:
    logger.info(
        f"Starting stream for {agent.name}"
        + (f" with min_tokens={stop_after_tokens}" if stop_after_tokens else " (full)")
    )
    token_count = 0
    last_logged = 0
    start_time = time.perf_counter()
    buf: list[str] = []
    async for chunk in agentsdk_text_stream(agent, prompt):
        inc = _count_tokens(chunk)
        token_count += inc
        if capture_text and isinstance(chunk, str) and chunk:
            buf.append(chunk)
        if token_count - last_logged >= log_every_tokens:
            if stop_after_tokens:
                logger.info(f"{agent.name}: {token_count}/{stop_after_tokens} tokens")
            else:
                logger.info(f"{agent.name}: {token_count} tokens streamed")
            last_logged = token_count
        if stop_after_tokens and token_count >= stop_after_tokens:
            logger.info(f"{agent.name}: Reached {token_count} tokens, stopping early")
            break
    elapsed = time.perf_counter() - start_time
    if stop_after_tokens:
        logger.info(
            f"{agent.name}: Preview complete - {token_count} tokens, {len(''.join(buf))} chars, {elapsed:.2f}s"
        )
    else:
        logger.info(f"Full answer streaming complete - approximately {token_count} tokens")
    return token_count, ("".join(buf).strip() if capture_text else None)

