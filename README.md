# servefit

**Before you rent the GPUs, check the map.** `servefit` inventories your hardware,
does the memory math for a model, checks a dated engine-support matrix, and tells
you what you can actually serve - and with what. Built for ML-infra folks who are
tired of "production inference engine" headlines that assume 64 Blackwell GPUs.

## 30 seconds

```bash
pip install git+https://github.com/techdevfun/servefit
servefit inventory
servefit fit --model Qwen/Qwen3-8B
```

![demo](examples/demo.gif)

## What it does

- `servefit inventory` - parse `nvidia-smi` (GPUs, VRAM, compute capability),
  CPU/RAM; AMD and Apple Silicon reported as detected-with-limits, never guessed.
- `servefit fit --model <hf-id-or-label>` - memory math (weights + KV cache at a
  stated context length, every assumption printed), viable engines from a curated
  matrix with reasons, one recommendation, and a starter config for that engine.
  States plainly which production components are still missing (load balancing,
  disaggregation) per engine.
- `servefit probe --endpoint <openai-compatible-url> --model <id>` - measure TTFT
  and tokens/sec against a running endpoint **on your hardware**.
- `--json out.json` on every command writes a full manifest.

Zero dependencies. Python 3.10+. Works offline with `--local <model-dir>` or
explicit `--params/--layers/--kv-heads`.

## Example

```
$ servefit fit --model Qwen/Qwen3-8B --from examples/profile-rtx4090.json

servefit fit: Qwen/Qwen3-8B  (family: qwen)

Memory math: weights 15.3 + KV 4.5 + overhead 1.0 = 20.8 GiB
  dtype bf16, ctx 32768, layers 36, kv-heads 8, head-dim 128

RECOMMENDED: SGLang
  + chip: compute_cap 8.9 >= 7.5
  + model: qwen family supported
  + memory: needs ~20.8 GiB, single-GPU VRAM has 24.0 GiB (fits)

Starter config:
  # install: pip install sglang[all]
  python -m sglang.launch_server --model-path Qwen/Qwen3-8B --context-length 32768

Missing for production (you still build these):
  - Smaller operator community than vLLM; fewer battle-tested deployment guides.
  - No built-in load balancing across replicas; bring your own gateway.
  - AMD/ROCm maturity trails NVIDIA.

Not viable:
  [no] DeepInfer
       chip: needs NVIDIA SM100 (Blackwell) or Ascend 950; have compute_cap 8.9
       model: qwen not in supported families (deepseek-v4.1-flash-only)
       ...

Assumptions:
  - weights = params x bytes-per-param (bf16 = 2.0 B)
  - kv cache = 2 x 36 layers x 8 kv-heads x 128 head-dim x 2.0 B x 32768 ctx ...
  - Matrix is a dated snapshot (see check dates); re-verify before spending money.
```

More samples in [`examples/`](examples/) (synthetic hardware profiles, labeled).

## The honest scope map

The engine matrix (`servefit/engines.py`) is curated and **dated** - every row
carries a check date and sources, and CI fails if any row goes 90 days without
re-verification. Headline claims get the honest treatment: DeepInfer's row, for
example, states its real scope (one model, two chip families, offline only, no
KV-cache system, no disaggregation) instead of repeating the "open-source
production inference engine" headline.

## What servefit is NOT

- It does not download models and does not provision GPUs.
- A "fits" verdict is not a production-readiness certificate. The README of your
  deployment still needs load balancing, autoscaling, and monitoring - servefit
  lists what's missing per engine instead of hiding it.
- Probe numbers are your machine's under your conditions, not a benchmark claim
  about the engine.
- The engine matrix is a snapshot. Hardware moves fast; every recommendation
  carries its check date. Re-verify before spending money.

## Install

```bash
pip install servefit        # PyPI (planned)
# or from source:
pip install .
```

## CLI reference

```
servefit inventory [--json out.json] [--from saved.json]
servefit fit --model <hf-id-or-label> [--local DIR] [--no-hf]
           [--params B] [--dtype bf16|fp16|fp32|fp8|int8|int4]
           [--context N] [--layers N] [--kv-heads N] [--head-dim N]
           [--traffic RPS] [--json out.json] [--from saved.json]
servefit probe --endpoint URL --model ID [--prompt TEXT]
           [--max-tokens N] [--json out.json]
servefit --version
```

`--from` loads a saved/synthetic inventory instead of detecting hardware
(useful for planning a purchase before the box exists).

## Roadmap

- Multi-GPU memory planner that suggests TP/PP splits, not just total pooling
- Concurrency-aware KV math (per-request sequences, not just single-sequence)
- Engine matrix coverage for more accelerators as they become verifiable
- `servefit compare --models a,b` side-by-side fit table

## License

Apache-2.0 - see [LICENSE](LICENSE).
