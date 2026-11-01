"""Curated engine-support matrix.

Every engine row carries a check_date and sources. The matrix is a snapshot,
not a live feed: hardware moves fast, and every recommendation carries its
check date so the user can re-verify before spending money.

Check dates below are the date this matrix was last curated by the
maintainers (ISO 8601). The CI matrix-integrity test fails if any row is
older than 90 days, forcing a re-verification instead of shipping stale.
"""

CHECK_DATE = "2026-11-01"

# Model-family keys used across the matrix. "other" means: verify support
# yourself before committing; the engine's docs are the source of truth.
MODEL_FAMILIES = ["llama", "qwen", "mistral", "deepseek", "gemma", "phi", "other"]

ENGINES = [
    {
        "name": "vLLM",
        "blurb": "High-throughput online serving with PagedAttention; the default pick for NVIDIA GPU serving.",
        "chips": ["nvidia-ampere+", "amd-rocm-experimental", "cpu"],
        "min_compute_cap": "7.0",
        "offline": True,
        "online": True,
        "quants": ["fp8", "awq", "gptq", "bitsandbytes", "marlin"],
        "model_families": ["llama", "qwen", "mistral", "deepseek", "gemma", "phi", "other"],
        "tensor_parallel": True,
        "missing_production": [
            "No built-in prefill/decode disaggregation (SGLang leads here).",
            "No built-in load balancing across replicas; bring nginx/HAProxy or a gateway.",
            "Multi-node tensor/pipeline parallelism is operationally heavy to run yourself.",
        ],
        "sources": ["https://docs.vllm.ai/", "https://github.com/vllm-project/vllm"],
        "check_date": CHECK_DATE,
    },
    {
        "name": "SGLang",
        "blurb": "Fast serving runtime with RadixAttention; first-class prefill/decode disaggregation.",
        "chips": ["nvidia-ampere+", "amd-rocm"],
        "min_compute_cap": "7.5",
        "offline": True,
        "online": True,
        "quants": ["fp8", "awq", "gptq"],
        "model_families": ["llama", "qwen", "mistral", "deepseek", "gemma", "phi", "other"],
        "tensor_parallel": True,
        "missing_production": [
            "Smaller operator community than vLLM; fewer battle-tested deployment guides.",
            "No built-in load balancing across replicas; bring your own gateway.",
            "AMD/ROCm maturity trails NVIDIA.",
        ],
        "sources": ["https://docs.sglang.ai/", "https://github.com/sgl-project/sglang"],
        "check_date": CHECK_DATE,
    },
    {
        "name": "TensorRT-LLM",
        "blurb": "NVIDIA's compiled inference: best raw latency on NVIDIA, at the cost of a build step.",
        "chips": ["nvidia-only"],
        "min_compute_cap": "7.5",
        "offline": True,
        "online": True,
        "quants": ["fp8", "int4-awq", "int8-sq"],
        "model_families": ["llama", "qwen", "mistral", "deepseek", "gemma", "phi"],
        "tensor_parallel": True,
        "missing_production": [
            "Every model+GPU combination needs an engine BUILD (compile/tune) step first.",
            "NVIDIA-only: no AMD, no Apple, no CPU path.",
            "Operational surface is larger (Triton or custom server, version pinning of TensorRT).",
        ],
        "sources": ["https://github.com/NVIDIA/TensorRT-LLM"],
        "check_date": CHECK_DATE,
    },
    {
        "name": "DeepInfer",
        "blurb": "DeepSeek's open-source inference engine. Narrow by design: one model, two chip families, offline only.",
        "chips": ["nvidia-sm100", "huawei-ascend-950"],
        "min_compute_cap": "10.0",
        "offline": True,
        "online": False,
        "quants": [],
        "model_families": ["deepseek-v4.1-flash-only"],
        "tensor_parallel": False,
        "missing_production": [
            "Single model: DeepSeek-V4.1-Flash only; your model almost certainly is not it.",
            "Two chip families only: NVIDIA SM100 (Blackwell, CUDA 13) or Huawei Ascend 950 (CANN 9.2).",
            "No KV-cache system, no prefill/decode disaggregation, no load balancing: "
            "the distance between 'can read the code' and 'runs production serving' is on you.",
        ],
        "sources": ["https://github.com/deepseek-ai/DeepInfer"],
        "check_date": CHECK_DATE,
    },
    {
        "name": "llama.cpp",
        "blurb": "Local inference everywhere: CPU, any GPU, Apple Silicon. GGUF models.",
        "chips": ["cpu", "nvidia-any", "amd-any", "apple-silicon"],
        "min_compute_cap": "",
        "offline": True,
        "online": True,
        "quants": ["gguf-q8", "gguf-q6", "gguf-q5", "gguf-q4"],
        "model_families": ["llama", "qwen", "mistral", "deepseek", "gemma", "phi", "other"],
        "tensor_parallel": False,
        "missing_production": [
            "Throughput is far below vLLM/SGLang on the same GPU; not a high-QPS serving engine.",
            "GGUF format only: your safetensors checkpoint needs conversion/quantization first.",
            "llama-server is a single-node server, not a serving tier.",
        ],
        "sources": ["https://github.com/ggerganov/llama.cpp"],
        "check_date": CHECK_DATE,
    },
    {
        "name": "Ollama",
        "blurb": "The easy button for local models: one command, bundled models, chat API.",
        "chips": ["cpu", "nvidia-any", "amd-any", "apple-silicon"],
        "min_compute_cap": "",
        "offline": True,
        "online": True,
        "quants": ["bundled"],
        "model_families": ["llama", "qwen", "mistral", "deepseek", "gemma", "phi", "other"],
        "tensor_parallel": False,
        "missing_production": [
            "Not a production serving engine: no batching for high QPS, no disaggregation.",
            "Model availability follows Ollama's library, not the full open-weight zoo.",
            "Single-node; multi-replica is on you.",
        ],
        "sources": ["https://ollama.com/", "https://github.com/ollama/ollama"],
        "check_date": CHECK_DATE,
    },
]


def get(name: str) -> dict | None:
    for e in ENGINES:
        if e["name"].lower() == name.lower():
            return e
    return None


def chip_supported(engine: dict, vendor: str, compute_cap: str) -> tuple:
    """Return (supported: bool, reason: str)."""
    chips = engine["chips"]
    if vendor == "nvidia":
        if "nvidia-only" in chips or "nvidia-ampere+" in chips:
            if engine["min_compute_cap"] and compute_cap:
                ok = compute_cap >= engine["min_compute_cap"]
                return ok, ("compute_cap %s >= %s" % (compute_cap, engine["min_compute_cap"])
                            if ok else
                            "needs compute_cap >= %s, have %s"
                            % (engine["min_compute_cap"], compute_cap))
            return True, "NVIDIA GPU detected"
        if "nvidia-sm100" in chips:
            ok = compute_cap == "10.0"
            return ok, ("SM100 (Blackwell) confirmed" if ok else
                        "needs NVIDIA SM100 (Blackwell) or Ascend 950; have compute_cap %s"
                        % (compute_cap or "unknown"))
        if "nvidia-any" in chips:
            return True, "any NVIDIA GPU works"
        return False, "engine has no NVIDIA path"
    if vendor == "amd":
        if any("rocm" in c or "amd" in c for c in chips):
            return True, "AMD via ROCm path (maturity varies by engine)"
        return False, "no AMD/ROCm path"
    if vendor == "apple":
        if "apple-silicon" in chips:
            return True, "Apple Silicon supported (unified memory)"
        return False, "no Apple Silicon path"
    if vendor == "unknown":
        return False, "GPU vendor unknown; not assumed"
    return False, "unsupported vendor"
