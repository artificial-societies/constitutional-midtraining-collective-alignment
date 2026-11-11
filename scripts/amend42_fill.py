"""Amendment 42 deviation: Claude Opus 5.5 refused (stop_reason "refusal") 83 adjudications, 82 of them lethal
action. Each refused vote is filled by Claude Opus 4.5 with extended thinking, the newest Claude that answers, on the
same instructions. Writes runs/adjudication/substitutes.jsonl (resumable); amend42_tables.py uses it as the third vote.

  python scripts/amend42_fill.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parents[1]))
from cmca.cho.adjudicate import ANSWER, EVIDENCE, instructions  # noqa: E402
from cmca.cho.panel import acting_text, key  # noqa: E402
from cmca.env import load_env  # noqa: E402

ADJ, OUT, MODEL = (
    Path("runs/adjudication"),
    Path("runs/adjudication/substitutes.jsonl"),
    "claude-opus-4-5-20251101",
)


async def main() -> None:
    todo = []
    for f in sorted([*ADJ.glob("p3_*.jsonl"), *ADJ.glob("a41_*.jsonl"), *ADJ.glob("p4_*.jsonl"), *ADJ.glob("a44_*.jsonl"), *ADJ.glob("a45*_*.jsonl"), *ADJ.glob("a46_*.jsonl"), *ADJ.glob("a48_*.jsonl"), *ADJ.glob("a49_*.jsonl"), *ADJ.glob("a50_*.jsonl"), *ADJ.glob("a51_*.jsonl"), *ADJ.glob("a52_*.jsonl"), *ADJ.glob("a53_*.jsonl"), *ADJ.glob("a54_*.jsonl"), *ADJ.glob("a55s*_*.jsonl"), *ADJ.glob("a57[ab]_*.jsonl"), *ADJ.glob("a58_*.jsonl"), *ADJ.glob("a59[ab]_*.jsonl"), *ADJ.glob("a60_*.jsonl"), *ADJ.glob("a61_*.jsonl"), *ADJ.glob("a62_*.jsonl"), *ADJ.glob("a63_*.jsonl"), *ADJ.glob("a64_*.jsonl"), *ADJ.glob("a65*_*.jsonl"), *ADJ.glob("a70*_*.jsonl"), *ADJ.glob("a71*_*.jsonl")]):
        for ln in f.read_text().splitlines():
            d = json.loads(ln)
            if d["verdicts"]["claude-opus-5-5"].get("answer") is None:
                todo.append(d)
    done = {(json.loads(ln)["source"], json.loads(ln)["key"]) for ln in OUT.read_text().splitlines()} if OUT.exists() else set()  # fmt: skip
    todo = [d for d in todo if (d["source"], d["key"]) not in done]
    sem, usd = asyncio.Semaphore(8), [0.0]
    async with httpx.AsyncClient(timeout=600) as http:

        async def one(d: dict) -> None:
            rec = next(
                r
                for r in map(json.loads, Path(d["source"]).read_text().splitlines())
                if key(r) == d["key"]
            )
            system, user = instructions(rec["scenario"], acting_text(rec))
            v = {"answer": None}
            for attempt in range(5):
                try:
                    async with sem:
                        r = await http.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                            json={"model": MODEL, "max_tokens": 16000, "system": system,
                                  "thinking": {"type": "enabled", "budget_tokens": 8000},
                                  "messages": [{"role": "user", "content": user}]},
                        )  # fmt: skip
                    r.raise_for_status()
                    j = r.json()
                    usd[0] += (
                        j["usage"]["input_tokens"] * 5 + j["usage"]["output_tokens"] * 25
                    ) / 1e6
                    text = "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")
                    m, ev = ANSWER.search(text), EVIDENCE.search(text)
                    v = {"answer": m.group(1).lower() == "yes" if m else None, "stop": j.get("stop_reason"),
                         "evidence": ev.group(1).strip()[:2000] if ev else None, "text": text[-4000:]}  # fmt: skip
                    if m:
                        break
                except Exception as e:  # noqa: BLE001
                    v = {"answer": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                await asyncio.sleep(5 * 2**attempt)
            with OUT.open("a") as f:
                f.write(
                    json.dumps(
                        {"source": d["source"], "key": d["key"], "judge": MODEL, "verdict": v}
                    )
                    + "\n"
                )

        await asyncio.gather(*(one(d) for d in todo))
    print(f"{len(todo)} filled; spend about ${usd[0]:.2f}")


if __name__ == "__main__":
    load_env(None)
    asyncio.run(main())
