"""Chat client with retries and cost accounting.

Agents and judges are served from one Azure AI Foundry resource through its OpenAI-compatible chat-completions
endpoint (AZURE_BASE_URL, AZURE_API_KEY). Claude models go through Anthropic's Messages API on the same
resource, and nemotron-* models through our own vLLM server on Modal (NEMOTRON_BASE_URL, NEMOTRON_API_KEY).
With CMCA_HOSTS=native, the OpenAI and Anthropic models in NATIVE go to their makers' own APIs instead
(OPENAI_API_KEY, ANTHROPIC_API_KEY), pinned to dated versions; every call records its host.
"""

import asyncio
import json
import os
import random
import time
from types import SimpleNamespace as NS
from typing import Any

import httpx2
import openai
from openai import AsyncOpenAI

type Message = dict[str, Any]

# $ per 1M tokens (input, output). Azure Foundry list prices as of 2026-10; estimates where marked.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-oss-120b": (0.15, 0.60),
    "DeepSeek-V3.2": (0.58, 1.68),  # estimate
    "Kimi-K2.6": (0.60, 2.50),  # estimate
    "DeepSeek-V4.1-Flash": (0.30, 1.20),  # estimate
    "Llama-4-Maverick-17B-128E-Instruct-FP8": (0.25, 1.00),
    "Mistral-Large-3": (0.50, 1.50),  # estimate
    "grok-4-1-fast-non-reasoning": (0.20, 0.50),
    "Cohere-command-a-plus-05-2026": (2.50, 10.0),  # estimate
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.4-swarm": (2.50, 15.0),  # conservative estimate
    "grok-4-20-non-reasoning": (3.00, 15.0),  # conservative estimate
    "DeepSeek-V4-Pro": (2.00, 4.00),  # conservative estimate
    "DeepSeek-V4-Flash-0731": (0.30, 1.20),
    "mistral-medium-3-5": (1.00, 4.00),
    "qwen3.8-27b": (0.30, 1.20),
    "qwen3-32b": (0.30, 1.20),
    "claude-opus-4-5": (5.00, 25.0),
    "nemotron-3-super": (0.0, 0.0),  # self-hosted on Modal: billed per GPU-hour, tracked separately
    "Llama-4-Scout-17B-16E-Instruct": (0.18, 0.60),
    "mistral-small-2503": (0.10, 0.30),
    "Phi-4": (0.13, 0.50),
    "claude-sonnet-4-5": (3.00, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-4-5": (1.00, 5.00),
}
DEFAULT_PRICE = (2.0, 8.0)

# Same models from their makers (CMCA_HOSTS=native), used while the Foundry resource was unavailable
# (2026-10-17); the model names in runs stay the Foundry names.
NATIVE: dict[str, tuple[str, str]] = {
    "gpt-5.4-swarm": ("openai", "gpt-5.4-2026-03-05"),
    "gpt-4.1": ("openai", "gpt-4.1-2025-04-14"),
    "gpt-4.1-mini": ("openai", "gpt-4.1-mini-2025-04-14"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini-2024-07-18"),
    "claude-haiku-4-5": ("anthropic", "claude-haiku-4-5-20251001"),
    "claude-opus-4-5": ("anthropic", "claude-opus-4-5-20251101"),
}


def native(model: str) -> tuple[str, str] | None:
    """(host, model id) when CMCA_HOSTS=native routes this model to its maker's API, else None."""
    return NATIVE.get(model) if os.environ.get("CMCA_HOSTS") == "native" else None


def self_hosted(model: str) -> bool:
    """Models served by our own vLLM server (NEMOTRON_BASE_URL): Nemotron, and the small models of the
    midtraining proof of concept (served as local-*)."""
    return model.startswith(("nemotron", "local-"))


def host(model: str) -> str:
    if (n := native(model)) is not None:
        return n[0]
    return "modal" if self_hosted(model) else "azure"


class Ledger:
    """Running spend across all calls of one invocation, with a hard cap."""

    def __init__(self, cap_usd: float) -> None:
        self.cap = cap_usd
        self.usd = 0.0
        self.calls = 0

    def add(self, model: str, prompt: int, completion: int) -> float:
        # self-hosted models are billed per GPU-hour, not per token
        pin, pout = (0.0, 0.0) if self_hosted(model) else PRICES.get(model, DEFAULT_PRICE)
        c = (prompt * pin + completion * pout) / 1e6
        self.usd += c
        self.calls += 1
        return c


class _AnthropicCompat:
    """Claude on Azure Foundry is served only through Anthropic's Messages API (/anthropic/v1/messages). This
    shim takes OpenAI-style chat.completions.create(**kw) calls (messages, tools, tool calls, tool results) and
    returns an OpenAI-shaped response. HTTP errors are re-raised as the matching openai exceptions so the
    callers' retry logic applies."""

    def __init__(self, url: str, key: str) -> None:
        self.url = url
        self.http = httpx2.AsyncClient(
            timeout=300, headers={"x-api-key": key, "anthropic-version": "2023-06-01"}
        )

    @staticmethod
    def _to_anthropic(messages: list[Message]) -> tuple[str, list[Message]]:
        system: list[str] = []
        out: list[Message] = []

        def add(role: str, blocks: list[Message]) -> None:
            if out and out[-1]["role"] == role:
                out[-1]["content"] += blocks
            else:
                out.append({"role": role, "content": blocks})

        for m in messages:
            c = m.get("content") or ""
            if m["role"] == "system":
                system.append(c)
            elif m["role"] == "user":
                add("user", [{"type": "text", "text": c or "(empty)"}])
            elif m["role"] == "tool":
                add(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": m["tool_call_id"],
                            "content": c or "(empty)",
                        }
                    ],
                )
            else:
                blocks: list[Message] = [{"type": "text", "text": c}] if c.strip() else []
                for tc in m.get("tool_calls") or []:
                    try:
                        inp = json.loads(tc["function"].get("arguments") or "{}")
                    except ValueError:
                        inp = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": inp if isinstance(inp, dict) else {},
                        }
                    )
                add("assistant", blocks or [{"type": "text", "text": "(empty)"}])
        return "\n\n".join(system), out

    async def create(self, **kw: Any) -> Any:
        system, msgs = self._to_anthropic(kw["messages"])
        body: Message = {
            "model": kw["model"],
            "messages": msgs,
            "max_tokens": kw.get("max_tokens") or kw.get("max_completion_tokens") or 8000,
        }
        if system:
            body["system"] = system
        if kw.get("temperature") is not None:
            body["temperature"] = kw["temperature"]
        if kw.get("tools"):
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters") or {"type": "object"},
                }
                for t in kw["tools"]
            ]
        try:
            r = await self.http.post(self.url, json=body)
        except httpx2.TimeoutException:
            raise openai.APITimeoutError(request=httpx2.Request("POST", self.url)) from None
        except httpx2.HTTPError:
            raise openai.APIConnectionError(request=httpx2.Request("POST", self.url)) from None
        if r.status_code != 200:
            fallback = openai.InternalServerError if r.status_code >= 500 else openai.APIStatusError
            err = {
                429: openai.RateLimitError,
                400: openai.BadRequestError,
                404: openai.NotFoundError,
            }.get(r.status_code, fallback)
            raise err(r.text[:500], response=r, body=None)
        d = r.json()
        text = "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
        tcs = [
            NS(
                id=b["id"],
                type="function",
                function=NS(name=b["name"], arguments=json.dumps(b["input"])),
            )
            for b in d["content"]
            if b["type"] == "tool_use"
        ]
        fin = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}.get(
            d.get("stop_reason"), "stop"
        )
        u = d.get("usage") or {}
        return NS(
            choices=[
                NS(
                    message=NS(content=text, tool_calls=tcs or None, model_extra={}),
                    finish_reason=fin,
                )
            ],
            usage=NS(
                prompt_tokens=u.get("input_tokens", 0), completion_tokens=u.get("output_tokens", 0)
            ),
        )


class Router:
    """Sends claude-* models to the Messages-API shim, nemotron-* models to the self-hosted vLLM server, and
    everything else to Azure chat-completions. Each backend is built on first use, so a run
    needs only the credentials of the backends it calls."""

    def __init__(self) -> None:
        self.azure: AsyncOpenAI | None = None
        self.anthropic: _AnthropicCompat | None = None
        self.nemotron: AsyncOpenAI | None = None
        self.openai_native: AsyncOpenAI | None = None
        self.anthropic_native: _AnthropicCompat | None = None

    async def create(self, **kw: Any) -> Any:
        m = kw["model"]
        if (n := native(m)) is not None:
            kw = {**kw, "model": n[1]}
            if n[0] == "openai":
                self.openai_native = self.openai_native or AsyncOpenAI(
                    api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=300
                )
                return await self.openai_native.chat.completions.create(**kw)
            self.anthropic_native = self.anthropic_native or _AnthropicCompat(
                "https://api.anthropic.com/v1/messages", os.environ["ANTHROPIC_API_KEY"]
            )
            return await self.anthropic_native.create(**kw)
        if m.startswith("claude"):
            self.anthropic = self.anthropic or _AnthropicCompat(
                azure_base().split("/openai")[0] + "/anthropic/v1/messages",
                os.environ["AZURE_API_KEY"],
            )
            return await self.anthropic.create(**kw)
        if self_hosted(m):
            if self.nemotron is None:
                self.nemotron = AsyncOpenAI(
                    base_url=os.environ["NEMOTRON_BASE_URL"].rstrip("/") + "/v1/",
                    api_key=os.environ["NEMOTRON_API_KEY"],
                    max_retries=0,
                    timeout=900,
                )
            kw.setdefault(
                "temperature", 1.0
            )  # NVIDIA's model card: temperature 1.0, top_p 0.95 for all tasks
            kw.setdefault("top_p", 0.95)
            return await self.nemotron.chat.completions.create(**kw)
        if self.azure is None:
            self.azure = AsyncOpenAI(
                base_url=azure_base() + "/",
                api_key=os.environ["AZURE_API_KEY"],
                max_retries=0,
                timeout=300,
            )
        return await self.azure.chat.completions.create(**kw)


def azure_base() -> str:
    base = os.environ["AZURE_BASE_URL"].rstrip("/")
    if not base.endswith("/openai/v1"):
        base = base.split("/openai")[0] + "/openai/v1"
    return base


class Client:
    """Chat calls with a concurrency limit, retries on transient errors, and spend charged to a Ledger."""

    def __init__(self, ledger: Ledger, concurrency: int = 32) -> None:
        self.router = Router()
        self.sem = asyncio.Semaphore(concurrency)
        self.ledger = ledger

    async def chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if self.ledger.usd >= self.ledger.cap:
            raise RuntimeError(f"budget cap ${self.ledger.cap} reached")
        kw: dict[str, Any] = {"model": model, "messages": messages}
        reasoning_oa = model.startswith(
            "gpt-5"
        )  # OpenAI reasoning deployments: no sampling knobs, max_completion_tokens
        if temperature is not None and not reasoning_oa:
            kw["temperature"] = temperature
        if top_p is not None and not reasoning_oa:
            kw["top_p"] = top_p
        if reasoning_effort:
            kw["reasoning_effort"] = reasoning_effort
        if max_tokens:
            kw["max_completion_tokens" if reasoning_oa else "max_tokens"] = max_tokens
        for attempt in range(12):
            try:
                async with self.sem:
                    t0 = time.monotonic()
                    r = await self.router.create(**kw)
                u = r.usage
                pt, ct = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
                usd = self.ledger.add(model, pt, ct)
                text = (r.choices[0].message.content or "") if r.choices else ""
                meta = {
                    "model": model,
                    "host": host(model),
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "usd": round(usd, 6),
                    "latency_s": round(time.monotonic() - t0, 2),
                }
                return text, meta
            except (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ):
                await asyncio.sleep(min(60, 2**attempt + random.random() * 2))
        raise RuntimeError(f"{model}: retries exhausted")
