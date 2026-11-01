"""servefit CLI: inventory, fit, probe. Stdlib only."""

from __future__ import annotations

import argparse
import datetime
import json
import sys

from servefit import __version__
from servefit import engines, fit as fitmod
from servefit import inventory as invmod
from servefit import memory_math, probe as probemod
from servefit import report


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.manifest(payload) + "\n")
    print("wrote %s" % path)


def cmd_inventory(args) -> int:
    if args.source:
        with open(args.source, encoding="utf-8") as f:
            inv = invmod.from_dict(json.load(f))
        print("# inventory loaded from %s (synthetic)" % args.source)
    else:
        inv = invmod.collect()
    print(report.hr_inventory(inv))
    if args.json:
        _write_json(args.json, {"generated_at": _utcnow(), "inventory": invmod.to_dict(inv)})
    return 0


def _resolve_arch(args) -> tuple:
    """Return (layers, kv_heads, head_dim, params_b, arch_notes)."""
    notes = []
    params_b = args.params
    cfg = None
    if args.local:
        cfg = memory_math.read_local_config(args.local)
        if cfg is None:
            print("warning: no config.json under %s" % args.local, file=sys.stderr)
    elif args.model and "/" in args.model and not args.no_hf:
        cfg = memory_math.fetch_hf_config(args.model)
    layers = args.layers
    kv_heads = args.kv_heads
    head_dim = args.head_dim
    hub_used = False
    if cfg is not None:
        cl, ck, ch, cp, cnotes = memory_math.arch_from_config(cfg)
        notes.extend(cnotes)
        layers = layers if layers is not None else cl
        kv_heads = kv_heads if kv_heads is not None else ck
        head_dim = head_dim if head_dim is not None else ch
        if params_b is None:
            params_b = cp
        hub_used = True
    if params_b is None and "/" in (args.model or "") and not args.no_hf:
        params_b = memory_math.fetch_hf_params_b(args.model)
        if params_b is not None:
            notes.append("params=%.2fB from Hugging Face Hub safetensors total" % params_b)
            hub_used = True
    if not hub_used and "/" in (args.model or "") and not args.no_hf:
        print("warning: HF Hub lookup failed (offline?); use --params/--layers/--kv-heads "
              "or --no-hf with explicit values", file=sys.stderr)
    if layers is None:
        layers = memory_math.ASSUMED_ARCH["layers"]
        notes.append("ASSUMED layers=%d (pass --layers to override)" % layers)
    if kv_heads is None:
        kv_heads = memory_math.ASSUMED_ARCH["kv_heads"]
        notes.append("ASSUMED kv_heads=%d (pass --kv-heads to override)" % kv_heads)
    if head_dim is None:
        head_dim = memory_math.ASSUMED_ARCH["head_dim"]
        notes.append("ASSUMED head_dim=%d (pass --head-dim to override)" % head_dim)
    if params_b is None:
        sys.exit("error: could not determine parameter count; pass --params (billions), "
                 "e.g. --params 8")
    return layers, kv_heads, head_dim, params_b, notes


def cmd_fit(args) -> int:
    if args.source:
        with open(args.source, encoding="utf-8") as f:
            inv = invmod.from_dict(json.load(f))
    else:
        inv = invmod.collect()
    layers, kv_heads, head_dim, params_b, arch_notes = _resolve_arch(args)
    mem = memory_math.total_gb(params_b, args.dtype, layers, kv_heads,
                               head_dim, args.context)
    mem["assumptions"].extend(arch_notes)
    result = fitmod.fit(args.model, mem, inv)
    print(report.hr_fit(result, mem))
    if args.json:
        _write_json(args.json, {
            "generated_at": _utcnow(),
            "inventory": invmod.to_dict(inv),
            "memory": mem,
            "fit": result,
        })
    return 0


def cmd_probe(args) -> int:
    res = probemod.probe_endpoint(args.endpoint, args.model, args.prompt,
                                  max_tokens=args.max_tokens)
    print(report.hr_probe(res))
    if args.json:
        _write_json(args.json, {"generated_at": _utcnow(), "probe": res})
    return 0 if res.get("ok") else 1


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="servefit",
        description="Honest scope map for serving LLMs on your own hardware: "
                    "inventory, memory math, engine fit, local throughput probe.")
    p.add_argument("--version", action="version", version="servefit " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inventory", help="inventory this machine's hardware")
    pi.add_argument("--json", metavar="FILE", help="also write a JSON manifest")
    pi.add_argument("--from", dest="source", metavar="FILE",
                    help="use a saved inventory JSON instead of detecting (synthetic)")
    pi.set_defaults(func=cmd_inventory)

    pf = sub.add_parser("fit", help="what fits on this box, with what engine")
    pf.add_argument("--model", required=True,
                    help="HF model id (e.g. Qwen/Qwen3-8B) or any label")
    pf.add_argument("--local", metavar="DIR",
                    help="local model dir with config.json (works offline)")
    pf.add_argument("--no-hf", action="store_true",
                    help="skip the Hugging Face Hub config lookup")
    pf.add_argument("--params", type=float, metavar="B",
                    help="parameter count in billions (overrides Hub/config)")
    pf.add_argument("--dtype", default="bf16",
                    choices=sorted(memory_math.DTYPE_BYTES),
                    help="weight dtype (default: bf16)")
    pf.add_argument("--context", type=int, default=32768, metavar="N",
                    help="context length for KV estimate (default: 32768)")
    pf.add_argument("--layers", type=int, help="transformer layers (overrides config)")
    pf.add_argument("--kv-heads", type=int, help="KV heads (overrides config)")
    pf.add_argument("--head-dim", type=int, help="head dim (overrides config)")
    pf.add_argument("--traffic", type=float, metavar="RPS",
                    help="expected requests/sec (informational; recorded in manifest)")
    pf.add_argument("--json", metavar="FILE", help="also write a JSON manifest")
    pf.add_argument("--from", dest="source", metavar="FILE",
                    help="use a saved inventory JSON instead of detecting (synthetic)")
    pf.set_defaults(func=cmd_fit)

    pp = sub.add_parser("probe", help="measure TTFT + tok/s against a running endpoint")
    pp.add_argument("--endpoint", required=True,
                    help="base URL of an OpenAI-compatible server")
    pp.add_argument("--model", required=True, help="model id the server serves")
    pp.add_argument("--prompt", default="Explain why the sky is blue in one sentence.",
                    help="prompt to send")
    pp.add_argument("--max-tokens", type=int, default=64)
    pp.add_argument("--json", metavar="FILE", help="also write a JSON manifest")
    pp.set_defaults(func=cmd_probe)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
