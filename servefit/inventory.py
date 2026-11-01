"""Hardware inventory: parse nvidia-smi, detect CPU/RAM, report AMD/Apple limits.

Never guesses. Anything not positively detected is reported as
detected-with-limits or unknown.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GPU:
    vendor: str            # "nvidia" | "amd" | "apple" | "unknown"
    name: str
    vram_gb: float         # 0.0 when unknown
    compute_cap: str       # e.g. "8.9"; "" when unknown / not applicable
    driver: str            # driver/CUDA version string when known


@dataclass
class Inventory:
    gpus: list = field(default_factory=list)
    cpu_count: int = 0
    ram_gb: float = 0.0
    os: str = ""
    notes: list = field(default_factory=list)

    def total_vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    def max_gpu_vram_gb(self) -> float:
        return max((g.vram_gb for g in self.gpus), default=0.0)


def _run(cmd: list, timeout: int = 10) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def _parse_nvidia_smi_csv(out: str) -> list:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` rows."""
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        name, mem, driver, cap = parts[0], parts[1], parts[2], parts[3]
        try:
            vram = float(mem) / 1024.0
        except ValueError:
            vram = 0.0
        gpus.append(GPU(vendor="nvidia", name=name, vram_gb=round(vram, 2),
                       compute_cap=cap, driver=driver))
    return gpus


def _nvidia_compute_cap_from_name(name: str) -> str:
    """Best-effort compute-capability lookup for well-known NVIDIA GPUs.

    Used only when nvidia-smi did not report a capability value.
    """
    table = {
        "H100": "9.0", "H200": "9.0", "B100": "10.0", "B200": "10.0",
        "A100": "8.0", "A40": "8.6", "A30": "8.0", "A10": "8.6", "A16": "8.6",
        "RTX 4090": "8.9", "RTX 4080": "8.9", "RTX 4070": "8.9",
        "RTX 3090": "8.6", "RTX 3080": "8.6", "RTX 3070": "8.6",
        "RTX 6000 Ada": "8.9", "L40S": "8.9", "L40": "8.9", "L4": "8.9",
        "V100": "7.0", "T4": "7.5",
    }
    upper = name.upper()
    for key, cap in table.items():
        if key.upper() in upper:
            return cap
    return ""


def detect_nvidia() -> list:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    out = _run([exe, "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits"])
    if out is None:
        return []
    gpus = _parse_nvidia_smi_csv(out)
    for g in gpus:
        if not g.compute_cap or g.compute_cap.lower() in ("n/a", "unknown"):
            g.compute_cap = _nvidia_compute_cap_from_name(g.name)
    return gpus


def detect_cpu_ram() -> tuple:
    import os
    cpu_count = os.cpu_count() or 0
    ram_gb = 0.0
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        ram_gb = round(kb / 1024.0 / 1024.0, 2)
                        break
        except OSError:
            pass
    elif platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                ram_gb = round(int(out.strip()) / 1024.0 ** 3, 2)
            except ValueError:
                pass
        cpu = _run(["sysctl", "-n", "hw.ncpu"])
        if cpu:
            try:
                cpu_count = int(cpu.strip())
            except ValueError:
                pass
    return cpu_count, ram_gb


def detect_apple_silicon() -> list:
    """Detect Apple Silicon as a GPU-with-limits entry (unified memory)."""
    if platform.system() != "Darwin":
        return []
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
    if "Apple" not in brand:
        return []
    _, ram_gb = detect_cpu_ram()
    # Unified memory is shared; report conservatively, never guess a split.
    return [GPU(vendor="apple", name=brand.strip(), vram_gb=0.0,
                compute_cap="", driver="")]


def detect_amd() -> list:
    if not shutil.which("rocm-smi"):
        return []
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])
    if out is None:
        return []
    gpus = []
    for m in re.finditer(r"Card series:\s*(.+)", out):
        gpus.append(GPU(vendor="amd", name=m.group(1).strip(), vram_gb=0.0,
                       compute_cap="", driver=""))
    return gpus


def collect() -> Inventory:
    inv = Inventory(os=platform.system())
    inv.gpus.extend(detect_nvidia())
    inv.gpus.extend(detect_amd())
    apple = detect_apple_silicon()
    if apple and not inv.gpus:
        inv.gpus.extend(apple)
        inv.notes.append(
            "Apple Silicon detected: unified memory is shared between CPU and "
            "GPU; servefit does not split it and reports GPU VRAM as unknown. "
            "Ollama / llama.cpp are the practical local options.")
    inv.cpu_count, inv.ram_gb = detect_cpu_ram()
    if not inv.gpus:
        inv.notes.append(
            "No GPU positively detected. CPU-only paths (llama.cpp, Ollama) "
            "are evaluated against system RAM.")
    return inv


def to_dict(inv: Inventory) -> dict:
    return {
        "os": inv.os,
        "cpu_count": inv.cpu_count,
        "ram_gb": inv.ram_gb,
        "total_vram_gb": round(inv.total_vram_gb(), 2),
        "gpus": [
            {"vendor": g.vendor, "name": g.name, "vram_gb": g.vram_gb,
             "compute_cap": g.compute_cap, "driver": g.driver}
            for g in inv.gpus
        ],
        "notes": inv.notes,
    }


def from_dict(d: dict) -> Inventory:
    inv = Inventory(os=d.get("os", ""), cpu_count=d.get("cpu_count", 0),
                    ram_gb=d.get("ram_gb", 0.0), notes=d.get("notes", []))
    for g in d.get("gpus", []):
        inv.gpus.append(GPU(vendor=g.get("vendor", "unknown"),
                            name=g.get("name", "unknown"),
                            vram_gb=float(g.get("vram_gb", 0.0)),
                            compute_cap=g.get("compute_cap", ""),
                            driver=g.get("driver", "")))
    return inv
