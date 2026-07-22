# Agent — Natural-Language Routing for Medical-Imaging CLI Tools

An AI agent and benchmarking suite that turns a plain-language request such as
*"segment the mandible on this CBCT scan"* into an executable command line for the
correct medical/dental imaging tool, with the correct parameters.

The agent works in two stages:

1. **Tool selection (router)** — pick the right CLI tool for the request.
2. **Parameter extraction** — pull each argument value out of the request, validate and
   type-convert it, then assemble a runnable command line.

Most of this repository is a **benchmark harness** that measures tool-selection accuracy
and parameter-extraction accuracy across several LLMs (local Ollama models and hosted
Claude) and several retrieval/RAG strategies (BM25, dense vector, hybrid, cross-encoder
reranker), evaluated over query variations: short/long, beginner/expert vocabulary,
missing-parameter cases, and multiple languages (English, French, Spanish, Portuguese).

The imaging tools themselves (the routing targets) are 3D Slicer-style CLI modules for
CBCT, intraoral-scan (IOS), and MRI processing.

---

## Repository layout

| Path | Purpose |
| --- | --- |
| `documentation/` | The tool catalog / knowledge base the agent retrieves over. `yaml/manifest.yaml` is the authoritative, machine-readable manifest describing each tool (name, path, description, tags, CLI style, typed parameter specs). `json/` holds lighter descriptions used by the notebooks and RAG (`without param.json`, `with example without param.json`, and `1 tool per file/` for per-tool embedding). |
| `queries/` | Ground-truth benchmark datasets. `Just tool/` holds `[query, expected_tool]` pairs; `Tool and param/` holds `{query, expected_tool, expected_params}` records in variants (short, long, beginners, expert vocab, normal vocab, missingparam, and french/spanish/portuguese). |
| `Just tool/` | Tool-selection-only experiment. `code/router.py` is a manifest-driven router that builds and runs the CLI; `code/compare_performance.py` benchmarks it (accuracy, latency, confidence). |
| `Just Parameters/` | Parameter-extraction-only experiment. `param.ipynb` extracts parameters for a given tool and scores per-parameter and perfect-match accuracy. |
| `Parameters and tools/` | The combined tool-selection + parameter-extraction study. `code/agent_router.py` is the two-stage select → extract → build-CLI runner; `parameter_extraction_improved.py` + `parameter_validator.py` do few-shot extraction with validation; `compare_all_models.py` / `compare_params_performance.py` drive the benchmarks; `score_claude_*.py` and `merge_*_into_comparison.py` score and merge Claude predictions. |
| `RAG/` | Retrieval experiment. `rag.ipynb` implements four retrieval modes (BM25, dense vector, hybrid, cross-encoder) with an optional LLM router stage. |
| `Full pipeline/` | End-to-end pipeline. `all.ipynb` / `benchmarked.py` run reranker → router → parameter agent and score reranker top-1/top-3, router accuracy, and parameter accuracy per model. |
| `CLI files/` | The real target CLI tools the agent routes to (ALI, AMASSS, AREG, ASO, MRI2CBCT, AutoCrop3D, DOCShapeAXI, MedX, …). These are external tools included as the routing corpus. |

---

## How it works

The router reads `documentation/yaml/manifest.yaml`, asks an LLM to choose a single tool
and (in the full pipeline) extract its parameters as structured JSON, validates the result
with `pydantic` schemas, then builds either a positional or flag-style command line
according to each tool's declared `cli_style`.

RAG variants first retrieve the most relevant tool descriptions (BM25, dense embeddings via
`mxbai-embed-large`, a hybrid of the two, or a `BAAI/bge-reranker-v2-m3` cross-encoder) and
feed the top-k candidates to the LLM before it decides.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally for the local models
  (e.g. `gemma`, `mistral`, `llama3`, `llama3.1:8b`, `qwen3:8b`, `qwen2.5-coder:7b`,
  `hermes3:8b`), plus the `mxbai-embed-large` embedding model.
- Python packages: `ollama`, `pydantic`, `PyYAML`, `rank_bm25`, `sentence-transformers`,
  `torch`, `numpy`, `tqdm`, `pandas`, `openpyxl`.
- Optional: a `llama.cpp` server (alternative local backend) and Anthropic Claude
  predictions for the Claude comparison.

```bash
python -m venv .venv && source .venv/bin/activate
pip install ollama pydantic PyYAML rank_bm25 sentence-transformers torch numpy tqdm pandas openpyxl
ollama pull gemma && ollama pull mxbai-embed-large   # plus any other models you want to benchmark
```

---

## Usage

**Route a single request (tool selection only):**

```bash
python "Just tool/code/router.py" "segment the mandible on this CBCT" --dry-run --verbose
```

**Full agent (select + extract parameters + build the command):**

```bash
python "Parameters and tools/code/agent_router.py" "register these two intraoral scans" --dry-run
```

Useful flags: `--dry-run` (print the command instead of running it), `--verbose`,
`--benchmark`, `--interactive`.

The router/agent read configuration from environment variables:
`ROUTER_MODEL`, `ROUTER_BACKEND` (`ollama` | `llamacpp`), `ROUTER_THRESHOLD`,
`PARAM_CONF_THRESHOLD`.

**Benchmarks:**

```bash
python "Just tool/code/compare_performance.py"
python "Parameters and tools/code/compare_all_models.py"
python "Full pipeline/benchmarked.py"
```

**Notebooks** (`RAG/rag.ipynb`, `Just Parameters/param.ipynb`, `Full pipeline/all.ipynb`):
set the config globals at the top of the notebook (model, query set, `top_k`, retrieval
mode) and run all cells.

> **Note on paths.** Some notebooks and scripts reference input paths (e.g.
> `input/documentation/tools`, `input/param/`) that predate the current folder names.
> Point them at `documentation/` and `queries/` before a clean run.

---

## Results

Benchmark outputs are written per experiment under each area's `results/` or `output/`
folder — per-query CSVs, aggregated JSON, `tool_selection_f1`, and Excel summaries for the
model comparisons (including local models, MedGemma, and Claude Opus / Sonnet).

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

The tools under `CLI files/` originate from their respective upstream projects and remain
subject to their own licenses.
