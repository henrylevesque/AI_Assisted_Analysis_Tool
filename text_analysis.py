import os
import sys
import time
import datetime
import re
import platform
import subprocess
from collections import Counter
from typing import Optional
import pandas as pd
import argparse
import json
import ollama
try:
    import yaml
except Exception:
    yaml = None
from tqdm import tqdm

__version__ = "1.3"

KNOWN_CONFIG_KEYS = {
    'models', 'input', 'output', 'id_col', 'content_col', 'prompt_desc',
    'runs', 'timeout', 'retries', 'delay',
    'within_model_consensus', 'within_model_consensus_mode', 'within_model_fuzzy_threshold',
    'between_model_consensus', 'between_model_consensus_mode', 'between_model_fuzzy_threshold',
    'aggregate', 'aggregated_consensus_mode', 'aggregate_consensus_mode',
    'aggregated_fuzzy_threshold', 'aggregate_fuzzy_threshold',
    'append_metadata',
}


def list_input_file(folder: str) -> Optional[str]:
    """Return the first CSV or XLSX file found in folder."""
    return next((f for f in os.listdir(folder) if f.endswith('.csv') or f.endswith('.xlsx')), None)


def _clean_path_helper(p: Optional[str]) -> Optional[str]:
    """Strip surrounding quotes and whitespace from a path string."""
    if p is None:
        return p
    p = str(p).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return p


def resolve_input_file(input_path: str) -> tuple[str, str]:
    """
    Resolve input to a single CSV or XLSX file.

    Returns (file_path, file_path). Raises ValueError or FileNotFoundError on bad input.
    """
    if not input_path:
        raise ValueError("Input path is required")

    input_path = _clean_path_helper(input_path)

    if os.path.isfile(input_path):
        if not (input_path.endswith('.csv') or input_path.endswith('.xlsx')):
            raise ValueError(f"Input file must be .csv or .xlsx, got: {input_path}")
        return input_path, input_path

    if os.path.isdir(input_path):
        found_file = list_input_file(input_path)
        if not found_file:
            raise FileNotFoundError(f"No .csv or .xlsx file found in folder: {input_path}")
        full_path = os.path.join(input_path, found_file)
        return full_path, full_path

    raise FileNotFoundError(f"Input path not found: {input_path}")


def resolve_output_path(input_file_path: str, output_spec: Optional[str], num_runs: int, num_models: int) -> str:
    """
    Resolve the output Excel file path from user-supplied output spec.

    Falls back to the input file's folder with an auto-generated name.
    Appends a timestamp if the resolved path already exists.
    """
    input_file_path = _clean_path_helper(input_file_path)
    output_spec = _clean_path_helper(output_spec) if output_spec else None

    def _add_ts_if_exists(path: str) -> str:
        if os.path.exists(path):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(path)
            return f"{base}_{ts}{ext}"
        return path

    if not output_spec:
        output_dir = os.path.dirname(input_file_path) or '.'
    elif os.path.isdir(output_spec):
        output_dir = output_spec
    elif output_spec.endswith(('.xlsx', '.csv')):
        return _add_ts_if_exists(output_spec)
    else:
        output_dir = output_spec
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    suffix = f"{num_runs}runs_multi.xlsx" if num_models > 1 else f"{num_runs}runs.xlsx"
    return _add_ts_if_exists(os.path.join(output_dir, f"{base_name}_text_analysis_{suffix}"))


def load_config(path: str) -> dict:
    """Load a JSON or YAML config file. Returns {} if path is falsy."""
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as fh:
        text = fh.read()
    try:
        return json.loads(text)
    except Exception:
        pass
    if yaml:
        try:
            return yaml.safe_load(text)
        except Exception:
            pass
    raise ValueError("Config file must be valid JSON or YAML")


def get_user_columns(df: pd.DataFrame) -> tuple[str, str, int]:
    """Interactively ask the user for column names and run count."""
    print('\nAvailable columns:')
    for idx, c in enumerate(df.columns):
        print(f'  {idx}: {c}')
    id_col = input('Identifier column name (Enter to use row numbers): ').strip()
    content_col = input('Text content column name (the column to analyze): ').strip()
    try:
        runs = int(input('How many times should each row be analyzed? [1]: ').strip() or '1')
    except Exception:
        runs = 1
    return id_col, content_col, runs


def _normalize_text(s: str) -> str:
    s = s or ''
    s = s.lower().strip()
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _split_and_normalize_set(s: str) -> list[str]:
    parts = re.split(r"[,;]+", s) if s else []
    normalized = [_normalize_text(p) for p in parts if _normalize_text(p)]
    seen: set[str] = set()
    out: list[str] = []
    for x in normalized:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _fuzzy_group_responses(responses: list[str], threshold: int = 85) -> list[list[str]]:
    try:
        from rapidfuzz import fuzz
    except Exception as e:
        raise RuntimeError("rapidfuzz is required for fuzzy mode; install it with: pip install rapidfuzz") from e

    groups: list[list[str]] = []
    for r in responses:
        placed = False
        for g in groups:
            if fuzz.token_set_ratio(g[0], r) >= threshold:
                g.append(r)
                placed = True
                break
        if not placed:
            groups.append([r])
    return groups


def compute_consensus_for_block(
    df_block: pd.DataFrame,
    response_cols: list[str],
    mode: str = 'exact',
    fuzzy_threshold: int = 85,
) -> tuple[list, list]:
    """
    Compute consensus and confidence for each row across the given response columns.

    Returns (consensus_list, confidence_list).
    """
    consensus: list = []
    confidences: list = []
    for _, row in df_block.iterrows():
        raw = [str(row[c]) for c in response_cols if pd.notna(row[c])]
        responses = [r for r in (r.strip() for r in raw) if r]
        if not responses:
            consensus.append('')
            confidences.append(0.0)
            continue
        if mode == 'exact':
            normalized = [_normalize_text(r) for r in responses]
            counts = Counter(normalized)
            most_common, count = counts.most_common(1)[0]
            consensus.append(most_common)
            confidences.append(round(count / len(responses), 3))
        elif mode == 'set':
            all_items: list[str] = []
            for r in responses:
                all_items.extend(_split_and_normalize_set(r))
            if not all_items:
                consensus.append('')
                confidences.append(0.0)
                continue
            counts = Counter(all_items)
            chosen = [it for it, ct in counts.items() if ct / len(responses) >= 0.5]
            if not chosen:
                chosen = [counts.most_common(1)[0][0]]
            consensus.append(', '.join(chosen))
            avg_freq = sum(counts[it] for it in chosen) / (len(chosen) * len(responses))
            confidences.append(round(avg_freq, 3))
        elif mode == 'fuzzy':
            normalized = [_normalize_text(r) for r in responses]
            groups = _fuzzy_group_responses(normalized, threshold=fuzzy_threshold)
            top = sorted(groups, key=lambda g: len(g), reverse=True)[0]
            rep, _ = Counter(top).most_common(1)[0]
            consensus.append(rep)
            confidences.append(round(len(top) / len(responses), 3))
        else:
            raise ValueError(f"Unknown consensus mode: {mode}")
    return consensus, confidences


def reorder_columns_posthoc(df_in: pd.DataFrame, models_list: list[str], num_runs: int) -> pd.DataFrame:
    """Reorder output columns: Identifier, Content, responses by run, consensus blocks, cross-model columns."""
    cols = ['Identifier', 'Content']
    for run_idx in range(1, num_runs + 1):
        for m in models_list:
            cols.append(f"Response_{run_idx} ({m})")
    for m in models_list:
        cols.append(f"Consensus ({m})")
        cols.append(f"Confidence ({m})")
    cols += ['Between_Consensus', 'Between_Confidence', 'Aggregate_Consensus', 'Aggregate_Confidence']
    existing = [c for c in cols if c in df_in.columns]
    remaining = [c for c in df_in.columns if c not in existing]
    return df_in[existing + remaining]


def _ollama_call(
    client: ollama.Client,
    model: str,
    prompt: str,
    max_retries: int = 2,
) -> str:
    """Call ollama.generate with exponential-backoff retry. Returns cleaned response text."""
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.generate(model=model, prompt=prompt)
            return resp['response'].strip().replace('\r', ' ').replace('\n', ' ')
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(2.0 * (2 ** attempt))
    return f"Error: {last_error}"


def run_model_on_texts(
    client: ollama.Client,
    model_name: str,
    texts: list,
    prompt_template: str,
    num_runs: int,
    ids: list,
    max_retries: int = 2,
) -> list[dict]:
    """Run a single model across all texts and return a list of response dicts."""
    rows = []
    for idx, txt in enumerate(tqdm(texts, desc=f"{model_name} texts", unit="row", file=sys.stdout, dynamic_ncols=True), 1):
        row_responses = []
        for _ in tqdm(range(num_runs), desc="runs", unit="run", leave=False, file=sys.stdout, total=num_runs, dynamic_ncols=True):
            result = _ollama_call(client, model_name, f"{prompt_template} {txt}", max_retries=max_retries)
            row_responses.append(result)
        row: dict = {'Identifier': ids[idx - 1]}
        for i, r in enumerate(row_responses, 1):
            row[f"Response_{i}"] = r
        rows.append(row)
    return rows


def detect_gpus(cpu_brand: str) -> str:
    """Return a string describing detected GPUs for the current platform."""
    gpu_info: Optional[str] = None
    integrated_gpu: Optional[str] = None
    for brand in ["Radeon", "NVIDIA", "Intel Graphics", "Iris", "GeForce", "RTX", "GTX"]:
        if brand.lower() in cpu_brand.lower():
            integrated_gpu = cpu_brand
            break
    try:
        if sys.platform.startswith('win'):
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True)
            gpus = [l.strip() for l in result.stdout.splitlines() if l.strip() and 'Name' not in l]
            gpu_info = ', '.join(gpus) if gpus else None
        elif sys.platform.startswith('linux'):
            r = subprocess.run('lspci | grep -i vga', shell=True, capture_output=True, text=True)
            gpus = [l for l in r.stdout.splitlines() if l]
            gpu_info = ', '.join(gpus) if gpus else None
            r3d = subprocess.run('lspci | grep -i 3d', shell=True, capture_output=True, text=True)
            gpus_3d = [l for l in r3d.stdout.splitlines() if l]
            if gpus_3d:
                gpu_info = (gpu_info + ', ' if gpu_info else '') + ', '.join(gpus_3d)
        elif sys.platform == 'darwin':
            r = subprocess.run(['system_profiler', 'SPDisplaysDataType', '-detailLevel', 'mini'], capture_output=True, text=True)
            gpus = [l.strip() for l in r.stdout.splitlines() if 'Chipset Model:' in l or 'Vendor:' in l]
            gpu_info = ', '.join(gpus) or None
    except Exception:
        gpu_info = None
    parts = []
    if integrated_gpu:
        parts.append(f"Integrated GPU: {integrated_gpu}")
    if gpu_info:
        parts.append(f"Detected GPU(s): {gpu_info}")
    return ', '.join(parts) if parts else 'Not detected'


def main() -> None:
    print('Python executable:', sys.executable)

    parser = argparse.ArgumentParser(description='Text analysis with multi-model comparisons and consensus')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--config', '-c', help='Path to JSON or YAML config file')
    parser.add_argument('--models', help='Comma-separated model names to run (overrides config)')
    parser.add_argument('--input', help='Input file or folder path')
    parser.add_argument('--output', help='Output file or folder path')
    parser.add_argument('--id-col', help='Identifier column name')
    parser.add_argument('--content-col', help='Content column name')
    parser.add_argument('--prompt-desc', help='What to identify in the text (e.g., "the main theme")')
    parser.add_argument('--runs', type=int, help='Number of runs per row')
    parser.add_argument('--timeout', type=float, help='Ollama request timeout in seconds [120]')
    parser.add_argument('--retries', type=int, help='Number of retries on Ollama failure [2]')

    grp_wm = parser.add_mutually_exclusive_group()
    grp_wm.add_argument('--within-model-consensus', dest='within_model_consensus', action='store_true')
    grp_wm.add_argument('--no-within-model-consensus', dest='within_model_consensus', action='store_false')
    parser.set_defaults(within_model_consensus=None)
    parser.add_argument('--within-model-consensus-mode', choices=['exact', 'set', 'fuzzy'])
    parser.add_argument('--within-model-fuzzy-threshold', type=int)

    grp_bm = parser.add_mutually_exclusive_group()
    grp_bm.add_argument('--between-model-consensus', dest='between_model_consensus', action='store_true')
    grp_bm.add_argument('--no-between-model-consensus', dest='between_model_consensus', action='store_false')
    parser.set_defaults(between_model_consensus=None)
    parser.add_argument('--between-model-consensus-mode', choices=['exact', 'set', 'fuzzy'])
    parser.add_argument('--between-model-fuzzy-threshold', type=int)

    parser.add_argument('--delay', type=float, help='Delay between model runs in seconds')

    grp_agg = parser.add_mutually_exclusive_group()
    grp_agg.add_argument('--aggregate', dest='aggregate', action='store_true')
    grp_agg.add_argument('--no-aggregate', dest='aggregate', action='store_false')
    parser.set_defaults(aggregate=None)

    grp_meta = parser.add_mutually_exclusive_group()
    grp_meta.add_argument('--append-metadata', dest='append_metadata', action='store_true')
    grp_meta.add_argument('--no-append-metadata', dest='append_metadata', action='store_false')
    parser.set_defaults(append_metadata=None)

    parser.add_argument('--no-interactive', action='store_true', help='Run non-interactively')
    args = parser.parse_args()

    cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(args.config) or {}
        except Exception as e:
            print(f"Could not load config {args.config}: {e}")
            if args.no_interactive:
                raise
        unknown = set(cfg.keys()) - KNOWN_CONFIG_KEYS
        if unknown:
            print(f"Warning: unknown config key(s): {', '.join(sorted(unknown))}. These will be ignored.")
        if not args.no_interactive:
            try:
                use_cfg = input(f"Load configuration from {args.config}? (y/n) [y]: ").strip().lower()
                if use_cfg not in ('', 'y', 'yes'):
                    cfg = {}
            except Exception:
                pass

    def _get(key: str, default=None):
        val = getattr(args, key.replace('-', '_'), None)
        if val is not None:
            return val
        if key in cfg:
            return cfg[key]
        key_u = key.replace('-', '_')
        if key_u in cfg:
            return cfg[key_u]
        return default

    # --- Model selection ---
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        discovered = [l.split()[0] for l in result.stdout.splitlines() if l.strip() and not l.lower().startswith("name")]
        if discovered:
            print("Available Ollama models:")
            for i, m in enumerate(discovered, 1):
                print(f"  {i}. {m}")
    except Exception:
        discovered = []

    suggested = ["gemma3:12b", "deepseek-r1:14b", "qwen3:14b"]
    cli_models = _get('models')
    models_to_run: list[str] = []
    if cli_models:
        models_to_run = [m.strip() for m in str(cli_models).split(',') if m.strip()]
    elif not args.no_interactive:
        multi = input("\nDo you want to run multiple models and compare results? (y/n): ").strip().lower() == 'y'
        if multi:
            print("Enter model names separated by commas (or press Enter to use all discovered/suggested models):")
            m_input = input().strip()
            models_to_run = [m.strip() for m in m_input.split(',') if m.strip()] if m_input else (discovered or suggested)
        else:
            single = input('Enter a model name (or press Enter for gemma3:latest): ').strip()
            models_to_run = [single or 'gemma3:latest']
    else:
        models_to_run = discovered or suggested

    # --- Input / output paths ---
    data_in = _get('input') or (input('\nPath to input file or folder [current directory]: ').strip() if not args.no_interactive else '.')
    data_out = _get('output') or (input('Path to output file or folder (Enter to save alongside input): ').strip() if not args.no_interactive else None)

    # --- Prompt description ---
    prompt_desc = _get('prompt_desc') or _get('prompt-desc') or None
    if not prompt_desc and not args.no_interactive:
        prompt_desc = input(
            '\nWhat should the model identify in each text?\n'
            '  (e.g., "the main theme", "the research methodology", "the geographic location")\n'
            '  > '
        ).strip() or None
    prompt_desc = prompt_desc or 'the main topic'

    data_in = _clean_path_helper(data_in)
    if data_out:
        data_out = _clean_path_helper(data_out)

    prompt_template = (
        f'I am going to give you a chunk of text. '
        f'Please identify {prompt_desc} used in the text. '
        f'Do not tell me anything besides {prompt_desc}. '
        f'If you tell me anything besides {prompt_desc} you will not be helpful. '
        f'The text is:'
    )

    try:
        input_file_path, _ = resolve_input_file(data_in)
    except (FileNotFoundError, ValueError) as e:
        print(f'Error: {e}')
        return

    print(f"\nLoading data from: {input_file_path}")
    df = pd.read_csv(input_file_path) if input_file_path.endswith('.csv') else pd.read_excel(input_file_path)

    # --- Column selection and run count ---
    id_col = _get('id_col') or None
    content_col = _get('content_col') or None
    num_runs: int = _get('runs') or 0
    if not (id_col and content_col and num_runs):
        if args.no_interactive:
            id_col = id_col or cfg.get('id_col')
            content_col = content_col or cfg.get('content_col')
            num_runs = num_runs or cfg.get('runs') or 1
        else:
            id_col, content_col, num_runs = get_user_columns(df)
    num_runs = num_runs or 1

    def _resolve_col_name(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        if name in df.columns:
            return name
        exact_ci = [c for c in df.columns if c.lower() == name.lower()]
        if exact_ci:
            if len(exact_ci) > 1:
                print(f"Warning: multiple columns match '{name}' case-insensitively. Using '{exact_ci[0]}'.")
            return exact_ci[0]
        starts = [c for c in df.columns if c.lower().startswith(name.lower())]
        if starts:
            print(f"Using column '{starts[0]}' for '{name}' (case-insensitive prefix match).")
            return starts[0]
        return None

    resolved_id = _resolve_col_name(id_col)
    resolved_content = _resolve_col_name(content_col)

    ids = df[resolved_id].tolist() if resolved_id else list(range(1, len(df) + 1))

    if not resolved_content:
        print(f"Content column '{content_col}' not found. Available: {list(df.columns)}. Exiting.")
        return
    content_col = resolved_content
    contents = df[content_col].tolist()

    # --- Delay between model switches ---
    switch_delay: float = _get('delay') or 0.0
    if len(models_to_run) > 1 and switch_delay == 0.0 and not args.no_interactive:
        delay_input = input("\nDelay in seconds between switching models (Enter for 0): ").strip()
        try:
            switch_delay = float(delay_input) if delay_input else 0.0
        except Exception:
            switch_delay = 0.0

    # --- Within-model consensus ---
    within_model = _get('within_model_consensus')
    if within_model is None:
        if args.no_interactive:
            # Auto-enable if more than one run; meaningless otherwise
            within_model = num_runs > 1
        elif num_runs <= 1:
            within_model = False
        else:
            within_model = input(
                f"\nCompute within-model consensus across the {num_runs} runs for each model? (y/n) [y]: "
            ).strip().lower() in ('', 'y', 'yes')
    else:
        within_model = within_model in (True, 'y', 'yes', 'Y', '1')

    within_model_mode: str = _get('within-model-consensus-mode') or ''
    within_model_fuzzy: int = _get('within-model-fuzzy-threshold') or 85
    if within_model and not within_model_mode and not args.no_interactive:
        mode_input = input(
            "  Consensus mode, exact (strict match), set (unordered lists), fuzzy (similar phrasing) [exact]:"
        ).strip().lower()
        within_model_mode = mode_input if mode_input in ('exact', 'set', 'fuzzy') else 'exact'
    within_model_mode = within_model_mode or 'exact'

    if within_model and within_model_mode == 'fuzzy' and not _get('within-model-fuzzy-threshold') and not args.no_interactive:
        thr = input("  Fuzzy similarity threshold, 0-100 (80-90 recommended) [85]: ").strip()
        try:
            within_model_fuzzy = int(thr) if thr else 85
        except Exception:
            within_model_fuzzy = 85

    # --- Between-model consensus (only meaningful with 2+ models) ---
    between_model = _get('between_model_consensus')
    if between_model is None:
        if args.no_interactive:
            between_model = len(models_to_run) > 1
        elif len(models_to_run) <= 1:
            between_model = False
        else:
            between_model = input(
                f"\nCompute between-model consensus across the {len(models_to_run)} per-model Consensus columns? (y/n) [y]: "
            ).strip().lower() in ('', 'y', 'yes')
    else:
        between_model = between_model in (True, 'y', 'yes', 'Y', '1')

    between_model_mode: str = _get('between-model-consensus-mode') or ''
    between_model_fuzzy: int = _get('between-model-fuzzy-threshold') or within_model_fuzzy
    if between_model and len(models_to_run) > 1 and not between_model_mode and not args.no_interactive:
        mode_input = input(
            f"  Consensus mode, exact, set, fuzzy [{within_model_mode}]:"
        ).strip().lower()
        between_model_mode = mode_input if mode_input in ('exact', 'set', 'fuzzy') else within_model_mode
    between_model_mode = between_model_mode or within_model_mode

    if between_model and between_model_mode == 'fuzzy' and not _get('between-model-fuzzy-threshold') and not args.no_interactive:
        thr_b = input(f"  Fuzzy threshold [{ between_model_fuzzy}]: ").strip()
        try:
            between_model_fuzzy = int(thr_b) if thr_b else between_model_fuzzy
        except Exception:
            pass

    # --- Aggregate consensus ---
    aggregated_choice = _get('aggregate')
    if aggregated_choice is None:
        if args.no_interactive:
            aggregated_choice = False
        else:
            aggregated_choice = input(
                "\nCompute aggregate consensus across all response columns? (y/n) [n]: "
            ).strip().lower() in ('y', 'yes')
    else:
        aggregated_choice = aggregated_choice in (True, 'y', 'yes', 'Y', '1')

    agg_mode: str = _get('aggregated_consensus_mode') or _get('aggregate_consensus_mode') or ''
    agg_fuzzy: int = _get('aggregated_fuzzy_threshold') or _get('aggregate_fuzzy_threshold') or within_model_fuzzy
    if aggregated_choice and not agg_mode and not args.no_interactive:
        mode_input = input(
            f"  Consensus mode, exact, set, fuzzy [{within_model_mode}]:"
        ).strip().lower()
        agg_mode = mode_input if mode_input in ('exact', 'set', 'fuzzy') else within_model_mode
    agg_mode = agg_mode or within_model_mode

    if aggregated_choice and agg_mode == 'fuzzy' and not _get('aggregated_fuzzy_threshold') and not args.no_interactive:
        thr_in = input(f"  Fuzzy threshold [{agg_fuzzy}]: ").strip()
        try:
            agg_fuzzy = int(thr_in) if thr_in else agg_fuzzy
        except Exception:
            pass

    # --- Append metadata ---
    append_metadata = _get('append_metadata')
    if append_metadata is None:
        append_metadata = True
    else:
        append_metadata = append_metadata in (True, 'y', 'yes', 'Y', '1')

    timeout_sec: float = _get('timeout') or 120.0
    max_retries: int = _get('retries') or 2

    # --- Analysis ---
    master_df = pd.DataFrame({'Identifier': ids, 'Content': contents})
    metadata_models: list[str] = []
    analysis_start = time.time()

    client = ollama.Client(timeout=timeout_sec)

    for idx, model in enumerate(models_to_run):
        print(f"\nRunning model: {model} ({idx + 1}/{len(models_to_run)})")
        metadata_models.append(model)
        rows = run_model_on_texts(client, model, contents, prompt_template, num_runs, ids, max_retries=max_retries)
        model_df = pd.DataFrame(rows)

        if 'Identifier' in model_df.columns:
            model_df = model_df.drop_duplicates(subset=['Identifier'])

        response_cols = [c for c in model_df.columns if c.lower().startswith('response')]
        renamed = {c: f"{c} ({model})" for c in response_cols}
        model_df = model_df.rename(columns=renamed)
        master_df = master_df.merge(model_df, on='Identifier', how='left')

        if within_model and response_cols:
            block_cols = [renamed[c] for c in response_cols]
            cons, confs = compute_consensus_for_block(master_df, block_cols, mode=within_model_mode, fuzzy_threshold=within_model_fuzzy)
            master_df[f"Consensus ({model})"] = cons
            master_df[f"Confidence ({model})"] = confs

        if idx < len(models_to_run) - 1 and switch_delay > 0:
            print(f"Waiting {switch_delay}s before next model...")
            time.sleep(switch_delay)

    wm_summary: dict = {}
    if within_model:
        for m in models_to_run:
            conf_col = f"Confidence ({m})"
            if conf_col in master_df.columns:
                s = master_df[conf_col].dropna()
                wm_summary[m] = {
                    'high': int((s >= 0.7).sum()),
                    'medium': int(((s >= 0.4) & (s < 0.7)).sum()),
                    'low': int((s < 0.4).sum()),
                }

    outpath = resolve_output_path(input_file_path, data_out, num_runs, len(models_to_run))
    output_dir = os.path.dirname(outpath) or '.'
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create output directory {output_dir}: {e}")

    try:
        master_df = reorder_columns_posthoc(master_df, models_to_run, num_runs)
    except Exception as e:
        print(f"Could not reorder columns: {e}")

    master_df.to_excel(outpath, index=False)
    analysis_end = time.time()
    analysis_duration = analysis_end - analysis_start
    print(f"\nAnalysis complete. Results saved to {outpath}")

    # --- Post-hoc: aggregate consensus (independent from between-model) ---
    try:
        try:
            df_report = pd.read_excel(outpath)
        except Exception as e:
            print(f"Could not read {outpath} for post-processing: {e}")
            df_report = None

        agg_summary: dict = {}
        ran_aggregated = False

        if aggregated_choice and df_report is not None:
            resp_cols = [col for col in df_report.columns if re.match(r'^response_\d+', col.lower())]
            print(f"\nFound {len(resp_cols)} response columns. Calculating aggregate consensus (mode={agg_mode})...")
            try:
                agg_cons, agg_conf = compute_consensus_for_block(df_report, resp_cols, mode=agg_mode, fuzzy_threshold=agg_fuzzy)
                df_report['Aggregate_Consensus'] = agg_cons
                df_report['Aggregate_Confidence'] = agg_conf
                ran_aggregated = True
            except Exception as e:
                print(f"Aggregate consensus failed: {e}")

            if ran_aggregated:
                high = df_report[df_report['Aggregate_Confidence'] >= 0.7]
                mid = df_report[(df_report['Aggregate_Confidence'] >= 0.4) & (df_report['Aggregate_Confidence'] < 0.7)]
                low = df_report[df_report['Aggregate_Confidence'] < 0.4]
                agg_summary = {'high': len(high), 'medium': len(mid), 'low': len(low), 'low_rows': low}
                try:
                    df_report.to_excel(outpath, index=False)
                    print(f"Aggregate consensus written to {outpath}")
                except Exception as e:
                    print(f"Could not write aggregate consensus: {e}")

        # --- Post-hoc: between-model consensus (independent) ---
        if between_model and len(metadata_models) > 1 and df_report is not None:
            cons_cols = [f"Consensus ({m})" for m in metadata_models if f"Consensus ({m})" in df_report.columns]
            if not cons_cols:
                cons_cols = [c for c in df_report.columns if c.lower().startswith('consensus (')]
            if cons_cols:
                print(f"Running between-model consensus on: {cons_cols}")
                try:
                    bm_cons, bm_conf = compute_consensus_for_block(df_report, cons_cols, mode=between_model_mode, fuzzy_threshold=between_model_fuzzy)
                    df_report['Between_Consensus'] = bm_cons
                    df_report['Between_Confidence'] = bm_conf
                    df_report.to_excel(outpath, index=False)
                    print(f"Between-model consensus saved to {outpath}")
                except Exception as e:
                    print(f"Between-model consensus failed: {e}")
            else:
                print("No per-model Consensus columns found for between-model consensus.")

        # --- Metadata ---
        if append_metadata:
            try:
                from openpyxl import load_workbook
                wb = load_workbook(outpath)
                ws = wb.active
                if ws is None:
                    raise ValueError("Could not access worksheet")
                ws.append([])
                ws.append(["Prompt used:", prompt_template])
                ws.append(["Models used:", ', '.join(metadata_models)])
                ws.append([f"Runs per row: {num_runs}"])
                ws.append([f"Delay between model runs: {switch_delay}s"])
                ws.append([f"Timeout: {timeout_sec}s, Retries: {max_retries}"])
                ws.append([f"Within-model consensus: {within_model} (mode: {within_model_mode})"])
                if within_model_mode == 'fuzzy':
                    ws.append([f"Within-model fuzzy threshold: {within_model_fuzzy}"])
                ws.append([f"Between-model consensus: {between_model} (mode: {between_model_mode})"])
                ws.append([f"Aggregate consensus: {aggregated_choice} (mode: {agg_mode})"])
                hours, rem = divmod(analysis_duration, 3600)
                minutes, seconds = divmod(rem, 60)
                ws.append([f"Duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s"])

                if wm_summary:
                    ws.append([])
                    ws.append(["Within-model confidence summary:"])
                    for m, counts in wm_summary.items():
                        ws.append([f"  {m}: high={counts['high']}, medium={counts['medium']}, low={counts['low']}"])

                if ran_aggregated:
                    ws.append([])
                    ws.append(["Aggregate consensus summary:"])
                    ws.append([f"  High confidence (>=70%): {agg_summary.get('high', 0)} rows"])
                    ws.append([f"  Medium confidence (40-69%): {agg_summary.get('medium', 0)} rows"])
                    ws.append([f"  Low confidence (<40%): {agg_summary.get('low', 0)} rows"])
                    low_rows = agg_summary.get('low_rows')
                    if agg_summary.get('low', 0) > 0 and low_rows is not None:
                        ws.append(["  Low-confidence rows (may need manual review):"])
                        for row_idx, row in low_rows.iterrows():
                            id_display = row.get('Identifier', row_idx + 1)
                            conf_val = row.get('Aggregate_Confidence', None)
                            conf_str = f"{conf_val:.1%}" if conf_val is not None else "unknown"
                            ws.append([f"    Row {row_idx + 1}: {id_display} (confidence: {conf_str})"])

                try:
                    import cpuinfo
                    cpu_brand = cpuinfo.get_cpu_info().get('brand_raw', 'Unknown CPU')
                except Exception:
                    cpu_brand = platform.processor() or platform.machine() or 'Unknown CPU'
                ws.append([f"CPU: {cpu_brand}"])
                ws.append([f"GPU: {detect_gpus(cpu_brand)}"])

                wb.save(outpath)
                print(f"Metadata appended to {outpath}")
            except Exception as e:
                print(f"Could not append metadata: {e}")

    except Exception as e:
        import traceback
        print(f"Post-processing error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
