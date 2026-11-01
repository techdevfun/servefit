"""Tests for servefit.memory_math (values hand-computed, see comments)."""

from servefit import memory_math as mm


def test_weights_8b_bf16():
    # 8e9 params x 2 B = 16e9 B / 2^30 = 14.901... GiB
    w, note = mm.weights_gb(8.0, "bf16")
    assert w == 14.9
    assert "bf16" in note


def test_weights_70b_fp8():
    # 70e9 x 1 B / 2^30 = 65.1918... GiB
    w, _ = mm.weights_gb(70.0, "fp8")
    assert w == 65.19


def test_kv_cache_qwen3_8b_32k():
    # 2 x 36 layers x 8 kv-heads x 128 head-dim x 2 B x 32768
    # = 4,831,838,208 B = 4.5 GiB
    k, note = mm.kv_cache_gb(36, 8, 128, "bf16", 32768)
    assert k == 4.5
    assert "32768" in note


def test_overhead_floor():
    o, _ = mm.overhead_gb(1.0, 0.5)
    assert o == 1.0  # floor, not 5% of 1.5


def test_overhead_scales():
    o, _ = mm.overhead_gb(100.0, 20.0)
    assert o == 6.0  # 5% of 120


def test_total_8b_bf16_32k():
    # weights 14.9 + kv 4.5 + overhead max(1, 0.05*19.4=0.97)=1.0 -> 20.4
    m = mm.total_gb(8.0, "bf16", 36, 8, 128, 32768)
    assert m["total_gb"] == 20.4
    assert len(m["assumptions"]) == 3


def test_dtype_table():
    assert mm.DTYPE_BYTES["int4"] == 0.5
    assert mm.DTYPE_BYTES["fp32"] == 4.0


def test_arch_from_config_full():
    cfg = {"num_hidden_layers": 36, "num_key_value_heads": 8,
           "num_attention_heads": 32, "hidden_size": 4096,
           "num_parameters": 8000000000}
    layers, kv, hd, params, notes = mm.arch_from_config(cfg)
    assert (layers, kv, hd) == (36, 8, 128)
    assert params == 8.0
    assert notes == []


def test_arch_from_config_sparse_uses_assumed_with_notes():
    cfg = {}
    layers, kv, hd, params, notes = mm.arch_from_config(cfg)
    assert (layers, kv, hd) == (32, 8, 128)
    assert params is None
    assert any("ASSUMED" in n for n in notes)


def test_read_local_config_missing_dir():
    assert mm.read_local_config("/nonexistent-dir-xyz") is None
