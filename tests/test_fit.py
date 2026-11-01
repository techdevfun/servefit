"""Tests for servefit.fit recommendation rules on synthetic inventories."""

from servefit import fit as fitmod
from servefit import inventory as invmod
from servefit import memory_math


def _rtx4090_box():
    inv = invmod.Inventory(os="Linux", cpu_count=16, ram_gb=64.0)
    inv.gpus.append(invmod.GPU("nvidia", "NVIDIA GeForce RTX 4090",
                               24.0, "8.9", "550.54"))
    return inv


def _mem_8b():
    return memory_math.total_gb(8.0, "bf16", 36, 8, 128, 32768)


def test_fit_8b_on_rtx4090_recommends_viable_engine():
    inv = _rtx4090_box()
    res = fitmod.fit("Qwen3-8B", _mem_8b(), inv)
    rec = res["recommendation"]
    assert rec is not None
    assert rec["engine"] in ("SGLang", "vLLM")  # preference order, both viable
    assert any("model" in line for line in res["viable"][0]["reasons"])
    # starter config is concrete and references the model
    assert len(rec["starter_config"]) >= 2
    assert any("Qwen3-8B" in c for c in rec["starter_config"])


def test_fit_8b_on_rtx4090_deepinfer_not_viable_with_chip_reason():
    inv = _rtx4090_box()
    res = fitmod.fit("Qwen3-8B", _mem_8b(), inv)
    names = [r["name"] for r in res["not_viable"]]
    assert "DeepInfer" in names
    di = [r for r in res["not_viable"] if r["name"] == "DeepInfer"][0]
    assert any("SM100" in r for r in di["reasons"])


def test_fit_70b_fp16_on_rtx4090_does_not_fit():
    inv = _rtx4090_box()
    mem = memory_math.total_gb(70.0, "fp16", 80, 8, 128, 32768)
    assert mem["total_gb"] > 24.0
    res = fitmod.fit("Llama-3.3-70B", mem, inv)
    vllm = [r for r in res["not_viable"] if r["name"] == "vLLM"][0]
    assert any("does NOT fit" in r for r in vllm["reasons"])


def test_fit_cpu_only_box_prefers_local_engines():
    inv = invmod.Inventory(os="Linux", cpu_count=8, ram_gb=64.0)
    mem = memory_math.total_gb(8.0, "bf16", 36, 8, 128, 8192)
    res = fitmod.fit("Qwen3-8B", mem, inv)
    names = [r["name"] for r in res["viable"]]
    assert "llama.cpp" in names or "Ollama" in names


def test_fit_two_a100_tensor_parallel_pool():
    inv = invmod.Inventory(os="Linux", cpu_count=32, ram_gb=128.0)
    for _ in range(4):
        inv.gpus.append(invmod.GPU("nvidia", "NVIDIA A100-SXM4-80GB",
                                   80.0, "8.0", "550"))
    mem = memory_math.total_gb(70.0, "fp16", 80, 8, 128, 32768)
    res = fitmod.fit("Llama-3.3-70B", mem, inv)
    vllm = [r for r in res["viable"] if r["name"] == "vLLM"]
    assert vllm, "70B should fit 2xA100 via tensor parallel"
    assert any("tensor-parallel" in r for r in vllm[0]["reasons"])


def test_model_family_detection():
    assert fitmod.model_family("Qwen/Qwen3-8B") == "qwen"
    assert fitmod.model_family("meta-llama/Llama-3.1-8B") == "llama"
    assert fitmod.model_family("some-org/custom-model") == "other"


def test_every_verdict_carries_reasons():
    inv = _rtx4090_box()
    res = fitmod.fit("Qwen3-8B", _mem_8b(), inv)
    for r in res["viable"] + res["not_viable"]:
        assert r["reasons"], r["name"]
        assert r["check_date"], r["name"]
