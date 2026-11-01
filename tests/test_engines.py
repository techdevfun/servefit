"""Tests for the curated engine matrix (servefit.engines)."""

import datetime

from servefit import engines as eng


def test_every_engine_has_check_date_within_90_days():
    today = datetime.date.today()
    for e in eng.ENGINES:
        d = datetime.date.fromisoformat(e["check_date"])
        age = (today - d).days
        assert 0 <= age <= 90, "%s check_date %s is stale (%d days)" % (e["name"], e["check_date"], age)


def test_every_engine_has_source_and_missing_components():
    for e in eng.ENGINES:
        assert e["sources"], e["name"]
        assert e["missing_production"], e["name"]
        assert e["blurb"], e["name"]


def test_deepinfer_requires_sm100():
    di = eng.get("DeepInfer")
    ok, why = eng.chip_supported(di, "nvidia", "8.9")
    assert not ok
    assert "SM100" in why
    ok2, _ = eng.chip_supported(di, "nvidia", "10.0")
    assert ok2


def test_deepinfer_single_model_scope():
    di = eng.get("DeepInfer")
    assert di["model_families"] == ["deepseek-v4.1-flash-only"]
    assert di["online"] is False


def test_vllm_compute_cap_gate():
    vllm = eng.get("vLLM")
    ok, _ = eng.chip_supported(vllm, "nvidia", "8.9")
    assert ok
    ok_old, why = eng.chip_supported(vllm, "nvidia", "6.1")
    assert not ok_old
    assert "6.1" in why


def test_llamacpp_accepts_cpu_and_apple():
    lc = eng.get("llama.cpp")
    ok, _ = eng.chip_supported(lc, "apple", "")
    assert ok
    assert "cpu" in lc["chips"]


def test_unknown_vendor_never_assumed():
    vllm = eng.get("vLLM")
    ok, _ = eng.chip_supported(vllm, "unknown", "")
    assert not ok
