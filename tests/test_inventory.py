"""Tests for servefit.inventory."""

from servefit import inventory as invmod


def test_parse_nvidia_smi_csv_two_gpus():
    out = ("NVIDIA RTX 4090, 24564, 550.54, 8.9\n"
           "NVIDIA A100-SXM4-40GB, 40960, 550.54, 8.0\n")
    gpus = invmod._parse_nvidia_smi_csv(out)
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA RTX 4090"
    assert gpus[0].vram_gb == round(24564 / 1024, 2)
    assert gpus[0].compute_cap == "8.9"
    assert gpus[1].compute_cap == "8.0"


def test_parse_nvidia_smi_csv_skips_blank_and_malformed():
    out = "\nNVIDIA T4, 15360, 535.54, 7.5\nnot,a,valid\n"
    gpus = invmod._parse_nvidia_smi_csv(out)
    assert len(gpus) == 1
    assert gpus[0].compute_cap == "7.5"


def test_parse_nvidia_smi_csv_unknown_vram_is_zero():
    gpus = invmod._parse_nvidia_smi_csv("Mystery GPU, N/A, 1.0, 8.9\n")
    assert gpus[0].vram_gb == 0.0


def test_compute_cap_name_lookup():
    assert invmod._nvidia_compute_cap_from_name("NVIDIA GeForce RTX 4090") == "8.9"
    assert invmod._nvidia_compute_cap_from_name("Tesla V100-PCIE-16GB") == "7.0"
    assert invmod._nvidia_compute_cap_from_name("Some Future GPU X9000") == ""


def test_dict_roundtrip():
    inv = invmod.Inventory(os="Linux", cpu_count=8, ram_gb=32.0,
                           notes=["n1"])
    inv.gpus.append(invmod.GPU("nvidia", "RTX 4090", 24.0, "8.9", "550"))
    d = invmod.to_dict(inv)
    back = invmod.from_dict(d)
    assert back.os == "Linux"
    assert back.cpu_count == 8
    assert back.gpus[0].vram_gb == 24.0
    assert back.notes == ["n1"]


def test_totals():
    inv = invmod.Inventory()
    inv.gpus.append(invmod.GPU("nvidia", "A", 24.0, "8.9", ""))
    inv.gpus.append(invmod.GPU("nvidia", "B", 24.0, "8.9", ""))
    assert inv.total_vram_gb() == 48.0
    assert inv.max_gpu_vram_gb() == 24.0


def test_collect_does_not_crash_and_never_guesses():
    inv = invmod.collect()
    assert inv.cpu_count >= 0
    for g in inv.gpus:
        assert g.vendor in ("nvidia", "amd", "apple", "unknown")


def test_amd_and_apple_detectors_empty_on_this_box():
    # This build box is x86_64 Linux without rocm-smi; assert graceful empties.
    assert invmod.detect_amd() == []
    assert invmod.detect_apple_silicon() == []
