"""Probe a running OpenAI-compatible endpoint: TTFT + tokens/sec, stdlib only.

Numbers are this machine's under current conditions, not a benchmark claim
about the engine. Labeled as such in every output.
"""

from __future__ import annotations

import json
import time
import urllib.request


def probe_endpoint(endpoint: str, model: str, prompt: str,
                   max_tokens: int = 64, timeout: int = 120) -> dict:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream",
                 "User-Agent": "servefit/0.1.0"})

    t_start = time.monotonic()
    ttft = None
    chunks = 0
    approx_tokens = 0
    done = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                chunks += 1
                if ttft is None:
                    ttft = time.monotonic() - t_start
                try:
                    obj = json.loads(payload)
                    delta = obj["choices"][0].get("delta", {}).get("content") or ""
                except (ValueError, KeyError, IndexError, TypeError):
                    delta = ""
                # Approximate: whitespace-split words, not true tokens.
                approx_tokens += len(delta.split())
    except Exception as exc:  # network errors are reported, not hidden
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    total = time.monotonic() - t_start
    decode_time = max(total - (ttft or 0.0), 1e-9)
    return {
        "ok": True,
        "endpoint": endpoint,
        "model": model,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(total, 3),
        "chunks": chunks,
        "approx_tokens": approx_tokens,
        "tokens_per_s": round(approx_tokens / decode_time, 1),
        "stream_done": done,
        "caveats": [
            "Numbers measured on THIS machine, under current load, against THIS endpoint.",
            "Token count is approximate (whitespace split), not the model's tokenizer.",
            "Not a benchmark claim about the engine.",
        ],
    }
