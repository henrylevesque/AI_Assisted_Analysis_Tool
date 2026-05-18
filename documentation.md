# AI Assisted Analysis Tool: Technical Documentation

This document is technical reference material for the AI Assisted Analysis Tool. It covers architecture, model selection, prompt design, the consensus algorithm, output format, and troubleshooting. For a quick-start guide and command reference, see [README.md](README.md). For ready-to-use prompt values, see [PROMPTS.md](PROMPTS.md).

---

## Table of Contents

1. [Model Selection and Performance](#model-selection-and-performance)
2. [Prompt Design](#prompt-design)
3. [Project Architecture](#project-architecture)
4. [Package Dependencies](#package-dependencies)
5. [Consensus Algorithm](#consensus-algorithm)
6. [Output Format](#output-format)
7. [Troubleshooting](#troubleshooting)
8. [Version History](#version-history)

---

## Model Selection and Performance

### Text Analysis

The Gemma family of models responds well to this tool's prompt structure. Gemma3 (12B) is the current recommended default for text analysis due to its strong instruction-following and reasonable speed on consumer hardware.

Models evaluated:

- **TinyLlama**: Too small; inconsistent and unreliable results.
- **Llama3.3**: High-quality outputs but too resource-intensive for batch processing.
- **Gemma2 (9B)**: Previously used default. Strong instruction-following, good throughput.
- **Gemma3 (12B)**: Current recommended default. Better instruction adherence than Gemma2, particularly for structured output tasks.
- **DeepSeek-R1 (14B)**: Performs well for analytical tasks; slower than Gemma3 on GPU-limited hardware.
- **Qwen3 (14B)**: Good results for structured extraction; comparable to Gemma3.

In general, using the smallest model that meets your quality bar gives the best performance trade-off. A 4B-parameter Gemma3 can give more consistent results with this tool's prompt structure than larger models from other labs.

### Image Analysis

Vision capability is generally available at larger parameter sizes. Recommended vision models:

- **llava:13b**: Reliable general-purpose vision; good balance of speed and quality.
- **llama3.2-vision:11b**: Strong instruction following for structured image tasks.
- **qwen2.5vl:7b**: Efficient option for simpler classification tasks.
- **gemma3:12b**: Vision-capable; performs well for design and spatial analysis.

Note: smaller vision models (7B and below) often struggle with complex scene descriptions or fine-grained object identification. If quality is critical, use 13B+ models.

---

## Prompt Design

### Structure

Both analysis scripts use a three-part prompt structure that has proven effective for local models, particularly the Gemma family:

**Text analysis prompt:**
```
I am going to give you a chunk of text.
Please identify {prompt_desc} used in the text.
Do not tell me anything besides {prompt_desc}.
If you tell me anything besides {prompt_desc} you will not be helpful.
The text is: {content}
```

**Image analysis prompt:**
```
You are a design expert in a design review. You will be shown an image.
Please tell me {type_of_analysis} concisely and only return {type_of_analysis}.
If multiple items are present, separate them with commas.
If you tell me anything other than {type_of_analysis}, you will not be helpful.
```

### Why this structure works

The prompt has three parts:
1. **Positive instruction**: what to return.
2. **Negative instruction**: what not to return.
3. **Helpfulness framing**: telling the model that off-task output makes it unhelpful.

The third part is particularly effective with Gemma variants and other RLHF-tuned models. These models are trained to be helpful assistants, so telling them that extra output is unhelpful suppresses preambles, explanations, and hedging.

### Prompt customization

The `prompt_desc` (text) and `type_of_analysis` (image) values are the only user-facing inputs to the prompt. See [PROMPTS.md](PROMPTS.md) for a curated library of values covering common research tasks.

---

## Project Architecture

### Core scripts

#### `text_analysis.py`
- **Input**: CSV or Excel file with an identifier column and a text content column.
- **Process**: Sends each row's content to one or more Ollama models, N times each (runs).
- **Output**: Excel file with all response columns plus optional consensus columns and metadata.

#### `image_analysis.py`
- **Input**: A single image file or a folder of images (JPG, PNG, BMP, TIFF, GIF, WebP).
- **Process**: Sends each image to one or more Ollama models, N times each (runs).
- **Output**: Excel file with response and consensus columns, and optional metadata.

### Configuration system

Settings are resolved in this priority order (highest wins):

```
CLI arguments  >  config file (YAML/JSON)  >  built-in defaults
```

Config files support both YAML and JSON formats. The `--config` flag loads a config file; `--no-interactive` suppresses all prompts so the run is fully automated.

### Ollama integration

Both scripts use `ollama.Client` with a configurable timeout (default: 120s) and call `client.generate()` (the raw completion endpoint, not `client.chat()`). This has lower overhead and avoids prepending a system/conversation context.

Failed calls are retried with exponential backoff (default: 2 retries, delays of 2s and 4s).

---

## Package Dependencies

```
ollama>=0.4.0        # Ollama Python client (generate API)
pandas>=2.0.0        # Data loading and manipulation
openpyxl>=3.1.0      # Excel read/write
tqdm>=4.60.0         # Progress bars
py-cpuinfo>=9.0.0    # CPU info for metadata reporting
rapidfuzz>=3.0.0     # Fuzzy string matching (optional; required for fuzzy consensus mode)
pyyaml>=6.0          # YAML config file parsing
```

### Virtual environment setup

```powershell
# Create and activate
python -m venv venv
.\venv\Scripts\activate.bat

# Install
pip install -r requirements.txt

# Pull a model
ollama pull gemma3:12b       # text analysis
ollama pull llava:13b        # image analysis
```

---

## Consensus Algorithm

### Overview

The tool computes up to three independent consensus types. Each type operates on a different set of columns and can be enabled or disabled independently.

| Type | Input columns | Output columns | Default |
|------|--------------|----------------|---------|
| Within-model | `Response_1 (m)` … `Response_N (m)` per model | `Consensus (m)`, `Confidence (m)` | ON when runs > 1 |
| Between-model | `Consensus (m1)`, `Consensus (m2)`, … | `Between_Consensus`, `Between_Confidence` | ON when 2+ models |
| Aggregate | All `Response_X (m)` columns | `Aggregate_Consensus`, `Aggregate_Confidence` | OFF |

### Consensus modes

#### Exact mode

Normalizes all responses (lowercase, strip punctuation, collapse whitespace) then picks the most frequent value. Best for short, discrete labels.

```
Confidence = count(most_frequent) / total_responses
```

#### Set mode

Splits each response on commas and semicolons, normalizes each token, then keeps items that appear in more than 50% of responses. Best for comma-separated lists (themes, keywords).

```
Confidence = mean(frequency_of_kept_items) / total_responses
```

#### Fuzzy mode

Groups responses using `rapidfuzz.fuzz.token_set_ratio`. Responses with similarity ≥ threshold are grouped together; the largest group wins. Best for free-text descriptions that may be paraphrased.

```
Confidence = size(largest_group) / total_responses
```

Recommended threshold: 80–90. Lower values merge more aggressively; higher values are stricter.

### Confidence levels

| Level | Range | Interpretation |
|-------|-------|---------------|
| High | ≥ 70% | Strong agreement; generally reliable |
| Medium | 40–69% | Moderate agreement; check borderline rows |
| Low | < 40% | Poor agreement; manual review recommended |

### Edge cases

- **Single response**: Confidence = 1.0 (trivially).
- **All empty responses**: Consensus = `''`, confidence = 0.0.
- **No majority in exact mode**: Most frequent value is used regardless of its share.
- **No items reach 50% threshold in set mode**: Most frequent item is used as fallback.

---

## Output Format

### Column ordering

Columns are written in this order:

1. `Identifier` / `Image` (row or image identifier)
2. `Content` (original text; text analysis only)
3. `Response_1 (model)` ... `Response_N (model)` (one column per run per model)
4. `Consensus (model)`, `Confidence (model)` (per-model within-model consensus)
5. `Between_Consensus`, `Between_Confidence` (cross-model consensus, if enabled)
6. `Aggregate_Consensus`, `Aggregate_Confidence` (aggregate consensus, if enabled)

### Metadata sheet

When `append_metadata: true`, the following is appended below the data:

- Prompt used
- Models used
- Runs per row/image
- Delay between model runs
- Timeout and retry settings
- Consensus settings (type, mode, threshold) for each enabled type
- Duration
- Aggregate confidence distribution (if aggregate was run)
- CPU and GPU information

### Reporting in Excel

To count and rank unique values in a consensus column:

```excel
=SORT(HSTACK(UNIQUE(A2:A100), COUNTIF(A2:A100, UNIQUE(A2:A100))), 2, -1)
```

This returns a two-column sorted table of (value, count) pairs that can be used to build a bar chart.

---

## Troubleshooting

### Ollama not found or not running

```powershell
# Start Ollama
ollama serve

# Verify a model is available
ollama list

# Pull a model if needed
ollama pull gemma3:12b
```

### Timeout errors

Increase `--timeout` (default: 120s). Vision models on CPU-only machines may need 300s or more.

```powershell
python image_analysis.py --config my_config.yaml --timeout 300 --no-interactive
```

### Model produces verbose or off-topic responses

1. Try a different model; Gemma3 and DeepSeek-R1 follow the prompt structure most reliably.
2. Use fuzzy consensus to tolerate minor variations in otherwise on-topic responses.
3. Increase runs; a wider sample gives more reliable consensus.

### Import errors

```powershell
# Ensure virtual environment is active
.\venv\Scripts\activate.bat

# Reinstall
pip install -r requirements.txt
```

### rapidfuzz not installed (fuzzy mode fails)

```powershell
pip install rapidfuzz
```

### Excel file locked

Close the output file in Excel before re-running. The script cannot write to a file that is open.

### PowerShell execution policy (Windows)

```powershell
.\venv\Scripts\activate.bat
# or
powershell -ExecutionPolicy Bypass -File .\venv\Scripts\Activate.ps1
```

### Performance tips

- Use smaller models for faster iteration (`gemma3:4b` instead of `gemma3:12b`).
- Start with `--runs 1` or `--runs 2` to test settings before a full run.
- Disable consensus types you don't need (`--no-between-model-consensus`, `--no-aggregate`).
- Use `exact` mode rather than `fuzzy` when responses are short and categorical (less compute).
- Process a small subset first: slice your input file to ~10 rows for testing.

---

## Version History

### v1.4 (2026-05-18)

- **Config key validation**: unknown keys in YAML/JSON config files now print a warning instead of being silently ignored. Helps catch typos like `ruuns: 5`.
- **Output file collision prevention**: auto-generated output filenames get a `_YYYYMMDD_HHMMSS` timestamp suffix if the path already exists. Explicit `.xlsx` paths also get a timestamp suffix on collision.
- **Model progress indicator**: the model loop now prints `(1/N)` / `(2/N)` so multi-model runs show progress in the terminal.
- **Within-model confidence summary in metadata**: the metadata sheet now reports high/medium/low confidence row counts per model for all within-model consensus runs, not just aggregate runs.
- **Em dashes removed from interactive prompts**: consensus mode selection prompts now use commas instead of em dashes.

### v1.3 (2025-05-18)

- **`ollama.generate` API**: switched from `chat` to `generate` (raw completion endpoint); lower overhead and more appropriate for single-turn structured extraction.
- **Timeout and retry**: configurable via `--timeout` (default: 120s) and `--retries` (default: 2); retries use exponential backoff.
- **Standardized column names**:
  - `Consensus_Confidence (model)` → `Confidence (model)`
  - `BetweenModel_Consensus` → `Between_Consensus`
  - `BetweenModel_Consensus_Confidence` → `Between_Confidence`
  - `Aggregated_Consensus` → `Aggregate_Consensus`
  - `Aggregated_Consensus_Confidence` → `Aggregate_Confidence`
- **Bug fix**: between-model consensus was only running when aggregate consensus was also enabled.
- **Bug fix**: `append_metadata` flag was not guarding the metadata block.
- **Bug fix**: within-model consensus was silently disabled for single-model runs regardless of run count (now correctly auto-enables when `runs > 1`).
- **Bug fix**: between-model consensus prompt was shown even when only one model was selected (image script).
- **New flags**: `--version`, `--prompt-desc` (text script), `--timeout`, `--retries`.
- **Interactive prompts**: reordered and reworded for consistency between text and image workflows; input/output paths and analysis description are now asked before consensus settings.
- **Removed**: `python_for_Zotero_abstracts/` scripts (superseded by `text_analysis.py` + `PROMPTS.md`), `archive/` legacy script.
- **Added**: `PROMPTS.md` prompt library.
- **CI**: GitHub Actions workflow now runs `run_local_consensus_test.py` on every push.
- **Dependencies**: minimum versions pinned in `requirements.txt`.

### v1.2-beta (2025-02-26)

- Unified `text_analysis.py` and `image_analysis.py` scripts replacing per-task scripts.
- Flexible input/output path resolution (file, folder, or auto-detect).
- Three independent consensus types: within-model, between-model, aggregated.
- YAML/JSON config file support with CLI override.
- Progress bars via `tqdm`.
- Metadata appended to Excel output.
