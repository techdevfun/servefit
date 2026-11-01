"""Tests for servefit.report."""

import json

from servefit import fit as fitmod
from servefit import inventory as invmod
from servefit import memory_math
from servefit import report


def _sample():
    inv = invmod.Inventory(os="Linux", cpu_count=16, ram_gb=64.0)
    inv.gpus.append(invmod.GPU("nvidia", "NVIDIA GeForce RTX 4090",
                               24.0, "8.9", "550.54"))
    mem = memory_math.total_gb(8.0, "bf16", 36, 8, 128, 32768)
    return fitmod.fit("Qwen3-8B", mem, inv), mem, inv


def test_hr_fit_shows_recommendation_and_limits():
    res, mem, _ = _sample()
    text = report.hr_fit(res, mem)
    assert "RECOMMENDED" in text
    assert "Memory math" in text
    assert "Assumptions" in text
    assert "Missing for production" in text


def test_hr_inventory_lists_gpu():
    _, _, inv = _sample()
    text = report.hr_inventory(inv)
    assert "RTX 4090" in text
    assert "24.0" in text


def test_manifest_is_valid_json():
    res, mem, inv = _sample()
    payload = {"fit": res, "memory": mem,
               "inventory": invmod.to_dict(inv)}
    assert json.loads(report.manifest(payload))["memory"]["total_gb"] == 20.4
