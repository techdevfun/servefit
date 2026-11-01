"""Human-readable tables + JSON manifests for servefit output."""

from __future__ import annotations

import json


def hr_inventory(inv) -> str:
    lines = ["Hardware inventory (%s)" % (inv.os or "unknown OS")]
    lines.append("  CPU: %d cores, RAM: %.1f GiB" % (inv.cpu_count, inv.ram_gb))
    if inv.gpus:
        for g in inv.gpus:
            lines.append("  GPU: %s | %s | %.1f GiB VRAM | compute_cap %s | driver %s"
                         % (g.vendor, g.name, g.vram_gb,
                            g.compute_cap or "unknown", g.driver or "unknown"))
    else:
        lines.append("  GPU: none detected")
    for n in inv.notes:
        lines.append("  note: " + n)
    return "\n".join(lines)


def hr_fit(fit_result: dict, mem: dict) -> str:
    L = []
    L.append("servefit fit: %s  (family: %s)" % (fit_result["model"], fit_result["model_family"]))
    L.append("")
    L.append("Memory math: weights %.1f + KV %.1f + overhead %.1f = %.1f GiB"
             % (mem["weights_gb"], mem["kv_cache_gb"], mem["overhead_gb"], mem["total_gb"]))
    L.append("  dtype %s, ctx %d, layers %d, kv-heads %d, head-dim %d"
             % (mem["dtype"], mem["context_len"], mem["layers"], mem["kv_heads"], mem["head_dim"]))
    L.append("")
    rec = fit_result["recommendation"]
    if rec:
        L.append("RECOMMENDED: %s" % rec["engine"])
        for r in rec["because"]:
            L.append("  + " + r)
        L.append("")
        L.append("Starter config:")
        for c in rec["starter_config"]:
            L.append("  " + c)
        L.append("")
        L.append("Missing for production (you still build these):")
        for m in rec["missing_production"]:
            L.append("  - " + m)
    else:
        L.append("RECOMMENDED: none (nothing viable on this box)")
    if fit_result["viable"]:
        L.append("")
        L.append("Viable engines:")
        for v in fit_result["viable"]:
            L.append("  [ok] %s  (matrix checked %s)" % (v["name"], v["check_date"]))
            for r in v["reasons"]:
                L.append("       " + r)
    if fit_result["not_viable"]:
        L.append("")
        L.append("Not viable:")
        for v in fit_result["not_viable"]:
            L.append("  [no] %s" % v["name"])
            for r in v["reasons"]:
                L.append("       " + r)
    L.append("")
    L.append("Assumptions:")
    for a in mem["assumptions"]:
        L.append("  - " + a)
    L.append("  - Matrix is a dated snapshot (see check dates); re-verify before spending money.")
    return "\n".join(L)


def hr_probe(res: dict) -> str:
    if not res.get("ok"):
        return "probe FAILED: %s" % res.get("error", "unknown error")
    L = ["Probe: %s  model %s" % (res["endpoint"], res["model"])]
    L.append("  TTFT: %s s" % res["ttft_s"])
    L.append("  Throughput: %s tok/s (approx, %d tokens in %s s)"
             % (res["tokens_per_s"], res["approx_tokens"], res["total_s"]))
    L.append("  stream completed: %s" % res["stream_done"])
    for c in res["caveats"]:
        L.append("  note: " + c)
    return "\n".join(L)


def manifest(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
