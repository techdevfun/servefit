"""Memory math for serving one model: weights + KV cache + overhead.

Every number carries its assumptions. Nothing is hidden in a formula.
"""

from __future__ import annotations

import json
import os
import urllib.request

GIB = 1024.0 ** 3

DTYPE_BYTES = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0,
    "int8": 1.0, "int4": 0.5,
}

# Fallback architecture facts when neither --layers/--kv-heads nor a
# config.json is available. Labeled as ASSUMED in every output.
ASSUMED_ARCH = {"layers": 32, "kv_heads": 8, "head_dim": 128}


def weights_gb(params_b: float, dtype: str) -> tuple:
    """Return (weights_gib, assumption_note)."""
    bpp = DTYPE_BYTES[dtype]
    gib = params_b * 1e9 * bpp / GIB
    return round(gib, 2), "weights = params x bytes-per-param (%s = %.1f B)" % (dtype, bpp)


def kv_cache_gb(layers: int, kv_heads: int, head_dim: int,
                dtype: str, context_len: int) -> tuple:
    """KV cache for one sequence: 2 (K+V) x layers x kv_heads x head_dim x bytes x ctx."""
    per_token = 2 * layers * kv_heads * head_dim * DTYPE_BYTES[dtype]
    gib = per_token * context_len / GIB
    return round(gib, 2), (
        "kv cache = 2 x %d layers x %d kv-heads x %d head-dim x %.1f B x %d ctx "
        "(single sequence; multiply by expected concurrency yourself)"
        % (layers, kv_heads, head_dim, DTYPE_BYTES[dtype], context_len))


def overhead_gb(weights: float, kv: float) -> tuple:
    """CUDA context + activations + allocator fragmentation estimate."""
    gib = max(1.0, 0.05 * (weights + kv))
    return round(gib, 2), ("overhead = max(1 GiB, 5%% of weights+kv); "
                           "covers CUDA context, activations, fragmentation")


def total_gb(params_b: float, dtype: str, layers: int, kv_heads: int,
             head_dim: int, context_len: int) -> dict:
    w, w_note = weights_gb(params_b, dtype)
    k, k_note = kv_cache_gb(layers, kv_heads, head_dim, dtype, context_len)
    o, o_note = overhead_gb(w, k)
    return {
        "params_b": params_b,
        "dtype": dtype,
        "context_len": context_len,
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "weights_gb": w,
        "kv_cache_gb": k,
        "overhead_gb": o,
        "total_gb": round(w + k + o, 2),
        "assumptions": [w_note, k_note, o_note],
    }


def fetch_hf_config(model_id: str, timeout: int = 10) -> dict | None:
    """Read config.json from the Hugging Face Hub (stdlib only)."""
    url = "https://huggingface.co/%s/resolve/main/config.json" % model_id.strip("/")
    req = urllib.request.Request(url, headers={"User-Agent": "servefit/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def fetch_hf_params_b(model_id: str, timeout: int = 10) -> float | None:
    """Read the safetensors parameter total from the HF models API."""
    url = "https://huggingface.co/api/models/%s" % model_id.strip("/")
    req = urllib.request.Request(url, headers={"User-Agent": "servefit/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        total = (d.get("safetensors") or {}).get("total")
        if total:
            return round(float(total) / 1e9, 2)
    except Exception:
        pass
    return None


def read_local_config(path: str) -> dict | None:
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def arch_from_config(cfg: dict) -> tuple:
    """Extract (layers, kv_heads, head_dim, params_b|None, notes) from a HF config."""
    notes = []
    layers = cfg.get("num_hidden_layers")
    kv_heads = cfg.get("num_key_value_heads")
    heads = cfg.get("num_attention_heads")
    hidden = cfg.get("hidden_size")
    head_dim = cfg.get("head_dim")
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads
    if layers is None:
        layers = ASSUMED_ARCH["layers"]
        notes.append("ASSUMED layers=%d (not in config)" % layers)
    if kv_heads is None:
        kv_heads = heads if heads else ASSUMED_ARCH["kv_heads"]
        notes.append("ASSUMED kv_heads=%d (not in config)" % kv_heads)
    if head_dim is None:
        head_dim = ASSUMED_ARCH["head_dim"]
        notes.append("ASSUMED head_dim=%d (not in config)" % head_dim)
    params_b = None
    for key in ("num_parameters", "n_params"):
        if cfg.get(key):
            params_b = float(cfg[key]) / 1e9
            break
    return layers, kv_heads, head_dim, params_b, notes
