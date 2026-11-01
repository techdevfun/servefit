"""Fit evaluation: which engines are viable on this box, and why.

Rules are deliberately boring and explicit: every verdict carries reasons,
and non-viable engines say exactly which requirement failed.
"""

from __future__ import annotations

from . import engines as eng
from . import inventory as invmod
from . import memory_math

# Preference order when several engines are viable: serving engines first,
# local engines as fallback tiers.
PREFERENCE = ["SGLang", "vLLM", "TensorRT-LLM", "DeepInfer", "llama.cpp", "Ollama"]


def model_family(model_id: str) -> str:
    lid = model_id.lower()
    for fam in ("deepseek", "llama", "qwen", "mistral", "gemma", "phi"):
        if fam in lid:
            return fam
    return "other"


def starter_config(engine_name: str, model_id: str, mem: dict,
                   inv: invmod.Inventory) -> list:
    """Starter command(s) for the recommended engine. Parse-able, copy-paste-able."""
    n_gpu = max(1, len(inv.gpus))
    ctx = mem["context_len"]
    if engine_name == "vLLM":
        tp = " --tensor-parallel-size %d" % n_gpu if n_gpu > 1 else ""
        return [
            "# install: pip install vllm   (needs a CUDA build matching your driver)",
            "vllm serve %s%s --max-model-len %d" % (model_id, tp, ctx),
        ]
    if engine_name == "SGLang":
        tp = " --tp %d" % n_gpu if n_gpu > 1 else ""
        return [
            "# install: pip install sglang[all]",
            "python -m sglang.launch_server --model-path %s%s --context-length %d"
            % (model_id, tp, ctx),
        ]
    if engine_name == "TensorRT-LLM":
        return [
            "# 1. build the engine for YOUR gpu (example: single GPU, fp16):",
            "trtllm-build --checkpoint_dir ./ckpt --output_dir ./engine "
            "--gemm_plugin float16 --max_batch_size 8 --max_input_len %d" % ctx,
            "# 2. serve it (see TensorRT-LLM docs for the Triton/server setup)",
        ]
    if engine_name == "DeepInfer":
        return [
            "# DeepInfer serves DeepSeek-V4.1-Flash on SM100 / Ascend 950 only.",
            "# Follow https://github.com/deepseek-ai/DeepInfer for the build steps.",
        ]
    if engine_name == "llama.cpp":
        return [
            "# 1. get a GGUF of the model (convert/quantize your safetensors first)",
            "llama-server -m model.gguf -c %d --n-gpu-layers 99" % ctx,
        ]
    if engine_name == "Ollama":
        return [
            "ollama pull %s   # if the model is in Ollama's library" % model_id,
            "ollama run %s" % model_id,
        ]
    return []


def evaluate_engine(engine: dict, mem: dict, inv: invmod.Inventory,
                    fam: str) -> dict:
    """Return {name, viable, reasons[], missing_production[], check_date}."""
    reasons = []
    viable = True

    # 1. chip support
    if inv.gpus:
        gpu = inv.gpus[0]
        ok, why = eng.chip_supported(engine, gpu.vendor, gpu.compute_cap)
        reasons.append("chip: " + why)
        if not ok:
            viable = False
    else:
        if "cpu" in engine["chips"]:
            reasons.append("chip: no GPU; CPU path available")
        else:
            reasons.append("chip: no GPU detected and engine has no CPU path")
            viable = False

    # 2. model support
    fams = engine["model_families"]
    if fam in fams:
        reasons.append("model: %s family supported" % fam)
    elif fam == "other":
        reasons.append("model: family not in curated list; verify in engine docs")
    else:
        reasons.append("model: %s not in supported families (%s)"
                       % (fam, ", ".join(fams)))
        viable = False

    # 3. memory fit
    need = mem["total_gb"]
    if inv.gpus and inv.gpus[0].vendor == "nvidia":
        pool = inv.total_vram_gb() if engine["tensor_parallel"] and len(inv.gpus) > 1 \
            else inv.max_gpu_vram_gb()
        pool_note = ("tensor-parallel across %d GPUs" % len(inv.gpus)
                     if engine["tensor_parallel"] and len(inv.gpus) > 1
                     else "single-GPU VRAM")
        if need <= pool * 0.95:
            reasons.append("memory: needs ~%.1f GiB, %s has %.1f GiB (fits)"
                           % (need, pool_note, pool))
        else:
            reasons.append("memory: needs ~%.1f GiB, %s only %.1f GiB (does NOT fit)"
                           % (need, pool_note, pool))
            viable = False
    elif inv.gpus and inv.gpus[0].vendor == "apple":
        reasons.append("memory: Apple unified memory is shared; servefit does not "
                       "split it. Verify the model fits your machine's total RAM "
                       "minus OS overhead yourself.")
    else:
        if inv.ram_gb and need <= inv.ram_gb * 0.8:
            reasons.append("memory: needs ~%.1f GiB, system RAM %.1f GiB (CPU path fits)"
                           % (need, inv.ram_gb))
        elif inv.ram_gb:
            reasons.append("memory: needs ~%.1f GiB, system RAM only %.1f GiB" % (need, inv.ram_gb))
            viable = False
        else:
            reasons.append("memory: RAM unknown; could not verify fit")
            viable = False

    # 4. online/offline is informational, not a gate (fit defaults to offline-capable)
    return {
        "name": engine["name"],
        "viable": viable,
        "reasons": reasons,
        "missing_production": engine["missing_production"],
        "check_date": engine["check_date"],
    }


def fit(model_id: str, mem: dict, inv: invmod.Inventory) -> dict:
    fam = model_family(model_id)
    results = [evaluate_engine(e, mem, inv, fam) for e in eng.ENGINES]
    viable = [r for r in results if r["viable"]]
    order = {name: i for i, name in enumerate(PREFERENCE)}
    viable.sort(key=lambda r: order.get(r["name"], 99))
    recommendation = None
    if viable:
        top = viable[0]
        recommendation = {
            "engine": top["name"],
            "because": top["reasons"],
            "starter_config": starter_config(top["name"], model_id, mem, inv),
            "missing_production": top["missing_production"],
        }
    return {
        "model": model_id,
        "model_family": fam,
        "recommendation": recommendation,
        "viable": viable,
        "not_viable": [r for r in results if not r["viable"]],
    }
