import os
import sys
import time
import re
import platform
import subprocess
from collections import Counter
import pandas as pd
import argparse
import json
try:
    import yaml
except Exception:
    yaml = None
from tqdm import tqdm
from ollama import generate

# Version for CLI --version
VERSION = "1.4"


def _call_generate_with_retries(model, prompt, images=None, timeout=120, retries=2):
    for attempt in range(retries + 1):
        try:
            if images is not None:
                resp = generate(model=model, prompt=prompt, images=images)
            else:
                resp = generate(model=model, prompt=prompt)
            return resp
        except Exception as e:
            if attempt >= retries:
                raise
            time.sleep(2 ** attempt)


def list_input_file(folder):
    """List first CSV or XLSX file found in folder."""
    return next((f for f in os.listdir(folder) if f.endswith('.csv') or f.endswith('.xlsx')), None)


def is_file(path):
    """Check if path is a file (not a directory)."""
    if not path:
        return False
    p = _clean_path_helper(path)
    return os.path.isfile(p)


def is_dir(path):
    """Check if path is a directory."""
    if not path:
        return False
    p = _clean_path_helper(path)
    return os.path.isdir(p)


def _clean_path_helper(p: str):
    """Helper to clean path (remove quotes, whitespace)."""
    if p is None:
        return p
    p = str(p).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return p


def resolve_input_file(input_path) -> tuple:
    """
    Resolve input to a single file (Excel or CSV).

    Args:
        input_path: Path to a file or folder

    Returns:
        (file_path, display_path) where file_path is the resolved file and display_path is for user messages

    Raises:
        FileNotFoundError: If file/folder cannot be found or no valid file in folder
    """
    if not input_path:
        raise ValueError("Input path is required")

    input_path = _clean_path_helper(input_path)

    # Case 1: Direct file path
    if os.path.isfile(input_path):
        if not (input_path.endswith('.csv') or input_path.endswith('.xlsx')):
            raise ValueError(f"Input file must be .csv or .xlsx, got: {input_path}")
        return input_path, input_path

    # Case 2: Directory path
    if os.path.isdir(input_path):
        found_file = list_input_file(input_path)
        if not found_file:
            raise FileNotFoundError(f"No .csv or .xlsx file found in folder: {input_path}")
        full_path = os.path.join(input_path, found_file)
        return full_path, full_path

    # Case 3: Path doesn't exist
    raise FileNotFoundError(f"Input path not found: {input_path}")


def resolve_output_path(input_file_path, output_spec, num_runs, num_models) -> str:
    """
    Resolve output file path from various output specifications.

    Args:
        input_file_path: The input file path (used for naming)
        output_spec: User-specified output (file path, folder path, or None)
        num_runs: Number of runs (for filename)
        num_models: Number of models (for filename)

    Returns:
        Full output file path ready to write to
    """
    input_file_path = _clean_path_helper(input_file_path)
    output_spec = _clean_path_helper(output_spec) if output_spec else None

    # Default: same folder as input file
    if not output_spec:
        output_dir = os.path.dirname(input_file_path) or '.'
    # Output spec is a directory
    elif os.path.isdir(output_spec):
        output_dir = output_spec
    # Output spec is a file path (with .xlsx or .csv extension) - use its directory and filename
    elif output_spec.endswith(('.xlsx', '.csv')):
        return output_spec
    # Output spec might be a folder that doesn't exist yet - create it
    else:
        output_dir = output_spec
        os.makedirs(output_dir, exist_ok=True)

    # Generate output filename from input filename
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    suffix = f"{num_runs}runs_multi.xlsx" if num_models > 1 else f"{num_runs}runs.xlsx"
    output_filename = f"{base_name}_text_analysis_{suffix}"

    return os.path.join(output_dir, output_filename)


def load_config(path: str):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as fh:
        text = fh.read()
    # try JSON first
    try:
        return json.loads(text)
    except Exception:
        pass
    # try YAML
    if yaml:
        try:
            return yaml.safe_load(text)
        except Exception:
            pass
    raise ValueError("Config file must be valid JSON or YAML")


def get_user_columns(df):
    print('\nAvailable columns:')
    for idx, c in enumerate(df.columns):
        print(f'  {idx}: {c}')
    id_col = input('Identifier column (name, Enter for auto): ').strip()
    content_col = input('Content column (name): ').strip()
    try:
        runs = int(input('Number of runs per row [1]: ').strip() or '1')
    except Exception:
        runs = 1
    return id_col, content_col, runs


def main():
    print('Python executable:', sys.executable)

    # --- CLI / Config handling ---
    parser = argparse.ArgumentParser(description='Text analysis with multi-model comparisons and consensus')
    parser.add_argument('--config', '-c', help='Path to JSON or YAML config file')
    parser.add_argument('--models', help='Comma-separated model names to run (overrides config)')
    parser.add_argument('--input', help='Input folder path')
    parser.add_argument('--prompt-desc', help='Short description of the prompt for metadata')
    parser.add_argument('--output', help='Output folder path')
    parser.add_argument('--id-col', help='Identifier column name')
    parser.add_argument('--content-col', help='Content column name')
    parser.add_argument('--runs', type=int, help='Number of runs per row')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout seconds for model calls (default:120)')
    parser.add_argument('--retries', type=int, default=2, help='Number of retries for model calls (default:2)')
    parser.add_argument('--version', action='store_true', help='Print version and exit')
    parser.add_argument('--save-every', type=int, default=50, help='Save interim results every N rows (default: 50)')
    # Consensus flags (within-model and between-model terminology)
    grp_wm = parser.add_mutually_exclusive_group()
    grp_wm.add_argument('--within-model-consensus', dest='within_model_consensus', action='store_true', help='Force within-model consensus on')
    grp_wm.add_argument('--no-within-model-consensus', dest='within_model_consensus', action='store_false', help='Force within-model consensus off')
    parser.set_defaults(within_model_consensus=None)
    parser.add_argument('--within-model-consensus-mode', choices=['exact', 'set', 'fuzzy'], help='Within-model consensus mode')
    parser.add_argument('--within-model-fuzzy-threshold', type=int, help='Within-model fuzzy threshold (0-100)')

    grp_bm = parser.add_mutually_exclusive_group()
    grp_bm.add_argument('--between-model-consensus', dest='between_model_consensus', action='store_true', help='Force between-model consensus on')
    grp_bm.add_argument('--no-between-model-consensus', dest='between_model_consensus', action='store_false', help='Force between-model consensus off')
    parser.set_defaults(between_model_consensus=None)
    parser.add_argument('--between-model-consensus-mode', choices=['exact', 'set', 'fuzzy'], help='Between-model consensus mode')
    parser.add_argument('--between-model-fuzzy-threshold', type=int, help='Between-model fuzzy threshold (0-100)')
    parser.add_argument('--delay', type=float, help='Delay between model runs in seconds')
    # Aggregation flag
    parser.add_argument('--aggregate', dest='aggregate', action='store_true', help='Aggregate AI responses for consensus across the output file')
    parser.add_argument('--no-aggregate', dest='aggregate', action='store_false', help='Do not aggregate AI responses for consensus across the output file')
    parser.set_defaults(aggregate=None)

    # Append metadata tri-state
    grp_meta = parser.add_mutually_exclusive_group()
    grp_meta.add_argument('--append-metadata', dest='append_metadata', action='store_true', help='Append metadata to output workbook')
    grp_meta.add_argument('--no-append-metadata', dest='append_metadata', action='store_false', help='Do not append metadata to output workbook')
    parser.set_defaults(append_metadata=None)
    parser.add_argument('--no-interactive', action='store_true', help='Run non-interactively (require args/config for prompts)')
    args = parser.parse_args()

    if getattr(args, 'version', False):
        print(VERSION)
        return

    cfg = {}
    if args.config:
        try:
            cfg = load_config(args.config) or {}
        except Exception as e:
            print(f"Could not load config {args.config}: {e}")
            if args.no_interactive:
                raise
            cfg = {}

    if args.config and not args.no_interactive:
        try:
            use_cfg = input(f"Load configuration from {args.config}? (y/n) [y]: ").strip().lower()
            if use_cfg not in ('', 'y', 'yes'):
                cfg = {}
        except Exception:
            pass

    def _get(key, default=None):
        val = getattr(args, key.replace('-', '_'), None)
        if val is not None:
            return val
        if key in cfg:
            return cfg[key]
        key_underscore = key.replace('-', '_')
        if key_underscore in cfg:
            return cfg[key_underscore]
        return default

    # Try to list available Ollama models
    try:
        models = []
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if line.strip() and not line.lower().startswith("name"):
                models.append(line.split()[0])
        if models:
            print("Available models:")
            for idx, m in enumerate(models, 1):
                print(f"  {idx}. {m}")
    except Exception:
        models = []

    suggested = ["gemma3:12b", "deepseek-ri:14b", "gpt-oss:20b"]
    cli_models = _get('models')
    models_to_run = []
    if cli_models:
        models_to_run = [m.strip() for m in str(cli_models).split(',') if m.strip()]
    else:
        if not args.no_interactive:
            multi = input("Do you want to compare multiple models? (y/n): ").strip().lower() == 'y'
            if multi:
                print("Enter model names to compare separated by commas, or press Enter to use suggested models:")
                m_input = input().strip()
                if not m_input:
                    models_to_run = models if models else suggested
                else:
                    models_to_run = [m.strip() for m in m_input.split(',') if m.strip()]
            else:
                single = input('Enter a single model to use (or press Enter for gemma2:latest): ').strip()
                single_model = single or 'gemma2:latest'
                if models and single_model not in models:
                    print(f"Warning: '{single_model}' not found in ollama list; proceeding with provided name.")
                models_to_run = [single_model]
        else:
            models_to_run = models if models else suggested

    data_in = _get('input') or (input('Data input file or folder [.] : ').strip() if not args.no_interactive else '.')
    data_out = _get('output') or (input('Data output file or folder [same as input] : ').strip() if not args.no_interactive else None)
    prompt_desc = _get('prompt_desc') or _get('prompt-desc') or None
    if not args.no_interactive:
        if data_in == '.':
            print("Tip: Using current directory ('.') as input. Specify --input <path> to use a different file/folder.")
        if not data_out:
            print("Tip: Output will be saved to the same folder as the input file.  Specify --output <path> to change.")
        if not prompt_desc:
            resp_desc = input('Enter what you want the model or models to identify within the text: ').strip()
            if resp_desc:
                prompt_desc = resp_desc
    prompt_desc = prompt_desc or 'the main topic'

    def _clean_path(p: str):
        return _clean_path_helper(p)

    data_in = _clean_path(data_in)
    if data_out:
        data_out = _clean_path(data_out)

    prompt_template = f'I am going to give you a chunk of text. Please identify {prompt_desc} used in the text. Do not tell me anything besides {prompt_desc}. If you tell me anything besides {prompt_desc} you will not be helpful. The text is:'

    # Resolve input file path
    try:
        input_file_path, _ = resolve_input_file(data_in)
    except (FileNotFoundError, ValueError) as e:
        print(f'Error: {e}')
        return

    # Load dataframe
    print(f"Loading data from: {input_file_path}")
    df = pd.read_csv(input_file_path) if input_file_path.endswith('.csv') else pd.read_excel(input_file_path)

    id_col = _get('id_col') or None
    content_col = _get('content_col') or None
    num_runs = _get('runs') or None
    if not (id_col and content_col and num_runs):
        if args.no_interactive:
            id_col = id_col or cfg.get('id_col')
            content_col = content_col or cfg.get('content_col')
            num_runs = num_runs or cfg.get('runs') or 1
        else:
            id_col, content_col, num_runs = get_user_columns(df)

    def _resolve_col_name(name):
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
            print(f"Using column '{starts[0]}' for requested '{name}' (case-insensitive startswith).")
            return starts[0]
        return None

    resolved_id = _resolve_col_name(id_col)
    resolved_content = _resolve_col_name(content_col)

    if resolved_id:
        ids = df[resolved_id].tolist()
    else:
        ids = list(range(1, len(df) + 1))

    if not resolved_content:
        print(f"Content column '{content_col}' not found (case-insensitive). Available columns: {list(df.columns)}. Exiting.")
        return
    content_col = resolved_content
    contents = df[content_col].tolist()

    # Resolve output path early so we can use it for interim saves
    outpath = resolve_output_path(input_file_path, data_out, num_runs, len(models_to_run))
    output_dir = os.path.dirname(outpath) or '.'
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create output directory {output_dir}: {e}")

    # How often to save interim results
    save_every = _get('save_every') or args.save_every or 50

    # Pre-run consensus and metadata choices
    within_model = _get('within_model_consensus') if _get('within_model_consensus') is not None else None
    if within_model is None:
        if args.no_interactive:
            within_model = True
        else:
            if len(models_to_run) == 1:
                within_model = False
            else:
                resp = input("Compute within-model consensus after runs? (y/n) [y]: ").strip().lower()
                within_model = True if resp in ('', 'y', 'yes') else False
    else:
        within_model = True if within_model in (True, 'y', 'yes', 'Y', '1') else False

    within_model_mode = _get('within-model-consensus-mode') or 'exact'
    within_model_fuzzy = _get('within-model-fuzzy-threshold') or 85
    if within_model and within_model_mode == 'fuzzy' and not _get('within-model-fuzzy-threshold') and not args.no_interactive:
        thr = input("Within-model fuzzy threshold (0-100) [85]: ").strip()
        try:
            within_model_fuzzy = int(thr) if thr else 85
        except Exception:
            within_model_fuzzy = 85

    aggregated_choice = _get('aggregate') if _get('aggregate') is not None else None
    if aggregated_choice is None:
        if args.no_interactive:
            aggregated_choice = False
        else:
            agg_resp = input("Do you want to aggregate AI responses for consensus across the output file? (y/n) [n]: ").strip().lower()
            aggregated_choice = True if agg_resp in ('y', 'yes') else False
    else:
        aggregated_choice = True if aggregated_choice in (True, 'y', 'yes', 'Y', '1') else False

    between_model = _get('between_model_consensus') if _get('between_model_consensus') is not None else None
    if between_model is None:
        if args.no_interactive:
            between_model = True
        else:
            if len(models_to_run) <= 1:
                between_model = False
            else:
                bm_resp = input("Compute between-model consensus across per-model Consensus columns? (y/n) [y]: ").strip().lower()
                between_model = True if bm_resp in ('', 'y', 'yes') else False
    else:
        between_model = True if between_model in (True, 'y', 'yes', 'Y', '1') else False

    between_model_mode = _get('between-model-consensus-mode') or within_model_mode
    between_model_fuzzy = _get('between-model-fuzzy-threshold') or within_model_fuzzy

    agg_mode = within_model_mode
    agg_fuzzy_thr = within_model_fuzzy
    if aggregated_choice:
        agg_mode = _get('aggregated_consensus_mode') or _get('aggregate_consensus_mode') or agg_mode
        agg_fuzzy_thr = _get('aggregated_fuzzy_threshold') or _get('aggregate_fuzzy_threshold') or agg_fuzzy_thr
        if not args.no_interactive:
            agg_mode = input(f"Aggregated consensus mode (exact/set/fuzzy) [{agg_mode}]: ").strip().lower() or agg_mode
            if agg_mode == 'fuzzy':
                thr_in = input(f"Aggregated fuzzy threshold (0-100) [{agg_fuzzy_thr}]: ").strip()
                try:
                    agg_fuzzy_thr = int(thr_in) if thr_in else agg_fuzzy_thr
                except Exception:
                    pass

    between_model_mode = _get('between-model-consensus-mode') or within_model_mode
    between_model_fuzzy = _get('between-model-fuzzy-threshold') or within_model_fuzzy
    if between_model and len(models_to_run) > 1 and not args.no_interactive:
        between_model_mode = input(f"Between-model consensus mode (exact/set/fuzzy) [{between_model_mode}]: ").strip().lower() or between_model_mode
        if between_model_mode == 'fuzzy':
            thr_b = input(f"Between-model fuzzy threshold (0-100) [{between_model_fuzzy}]: ").strip()
            try:
                between_model_fuzzy = int(thr_b) if thr_b else between_model_fuzzy
            except Exception:
                pass

    append_metadata = _get('append_metadata') if _get('append_metadata') is not None else None
    if append_metadata is None:
        append_metadata = True
    else:
        append_metadata = True if append_metadata in (True, 'y', 'yes', 'Y', '1') else False

    switch_delay = _get('delay') if _get('delay') is not None else 0.0
    if switch_delay is None:
        switch_delay = 0.0
    if len(models_to_run) > 1 and switch_delay == 0.0 and not args.no_interactive:
        delay_input = input("Enter delay between model runs in seconds (e.g., 1.0) or press Enter for 0: ").strip()
        switch_delay = float(delay_input) if delay_input else 0.0

    # Master dataframe
    master_df = pd.DataFrame({'Identifier': ids, 'Content': contents})
    metadata_models = []
    analysis_start = time.time()

    def _normalize_text(s: str) -> str:
        s = s or ''
        s = s.lower().strip()
        s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def _split_and_normalize_set(s: str):
        parts = re.split(r"[,;]+", s) if s else []
        normalized = [_normalize_text(p) for p in parts if _normalize_text(p)]
        seen = set()
        out = []
        for x in normalized:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _fuzzy_group_responses(responses, threshold=85):
        try:
            from rapidfuzz import fuzz
        except Exception as e:
            raise RuntimeError("rapidfuzz is required for fuzzy grouping; please install it (pip install rapidfuzz)") from e

        groups = []
        for r in responses:
            placed = False
            for g in groups:
                rep = g[0]
                score = fuzz.token_set_ratio(rep, r)
                if score >= threshold:
                    g.append(r)
                    placed = True
                    break
            if not placed:
                groups.append([r])
        return groups

    def compute_consensus_for_block(df_block, response_cols, mode='exact', fuzzy_threshold=85):
        consensus = []
        confidences = []
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
                conf = count / len(responses)
                consensus.append(most_common)
                confidences.append(round(conf, 3))
            elif mode == 'set':
                all_items = []
                for r in responses:
                    items = _split_and_normalize_set(r)
                    all_items.extend(items)
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
                groups_sorted = sorted(groups, key=lambda g: len(g), reverse=True)
                top = groups_sorted[0]
                counts = Counter(top)
                rep, count = counts.most_common(1)[0]
                conf = len(top) / len(responses)
                consensus.append(rep)
                confidences.append(round(conf, 3))
            else:
                raise ValueError(f"Unknown consensus mode: {mode}")
        return consensus, confidences

    def reorder_columns_posthoc(df_in, models_list, num_runs):
        cols = ['Identifier', 'Content']
        for run_idx in range(1, num_runs + 1):
            for m in models_list:
                cols.append(f"Response_{run_idx} ({m})")
        for m in models_list:
            cols.append(f"Consensus ({m})")
            cols.append(f"Consensus_Confidence ({m})")
        existing = [c for c in cols if c in df_in.columns]
        remaining = [c for c in df_in.columns if c not in existing]
        final_order = existing + remaining
        return df_in[final_order]

    def run_model_on_texts(model_name, texts, prompt_template, num_runs, ids, outpath, save_every=50, timeout=120, retries=2):
        """
        Run model on texts using stateless generate calls (no accumulated context).
        Saves interim results every save_every rows to protect against crashes.
        """
        rows = []
        for idx, txt in enumerate(tqdm(texts, desc=f"{model_name} texts", unit="row", file=sys.stdout, dynamic_ncols=True), 1):
            row_responses = []
            # Skip empty or NaN content gracefully
            if not txt or (isinstance(txt, float)):
                for i in range(1, num_runs + 1):
                    row_responses.append('')
            else:
                for run in tqdm(range(num_runs), desc="runs", unit="run", leave=False, file=sys.stdout, total=num_runs, dynamic_ncols=True):
                    try:
                        # Use generate (stateless) instead of chat to prevent context accumulation
                        resp = _call_generate_with_retries(model=model_name, prompt=f"{prompt_template} {txt}", timeout=timeout, retries=retries)
                        cleaned = resp['response'].strip().replace('\r', ' ').replace('\n', ' ')
                        row_responses.append(cleaned)
                    except Exception as e:
                        row_responses.append(f"Error: {e}")
            result = {'Identifier': ids[idx - 1]}
            for i, r in enumerate(row_responses, 1):
                result[f"Response_{i}"] = r
            rows.append(result)

            # Incremental save every save_every rows
            if idx % save_every == 0:
                try:
                    interim_df = pd.DataFrame(rows)
                    interim_path = outpath.replace('.xlsx', f'_interim_{idx}rows.xlsx')
                    interim_df.to_excel(interim_path, index=False)
                    print(f"\n  [Checkpoint] Saved {idx} rows to {interim_path}")
                except Exception as e:
                    print(f"\n  [Checkpoint] Could not save interim at row {idx}: {e}")

        return rows

    # Run models sequentially
    for idx, m in enumerate(models_to_run):
        model = m
        print(f"\nRunning model: {model}")
        metadata_models.append(model)
        timeout_val = _get('timeout') or 120
        retries_val = _get('retries') or 2
        prompt_desc = _get('prompt-desc') or None
        rows = run_model_on_texts(model, contents, prompt_template, num_runs, ids, outpath, save_every=save_every, timeout=timeout_val, retries=retries_val)
        model_df = pd.DataFrame(rows)

        # Drop duplicate Identifier rows before merging
        if 'Identifier' in model_df.columns:
            model_df = model_df.drop_duplicates(subset=['Identifier'])

        # Rename response columns to include model name
        response_cols = [c for c in model_df.columns if c.lower().startswith('response')]
        renamed = {c: f"{c} ({model})" for c in response_cols}
        model_df = model_df.rename(columns=renamed)

        # Merge responses into master_df by Identifier
        master_df = master_df.merge(model_df, on='Identifier', how='left')

        # Compute within-model consensus if requested
        if within_model and response_cols:
            block_cols = [renamed[c] for c in response_cols]
            consensus, confidences = compute_consensus_for_block(master_df, block_cols, mode=within_model_mode, fuzzy_threshold=within_model_fuzzy)
            master_df[f"Consensus ({model})"] = consensus
            master_df[f"Consensus_Confidence ({model})"] = confidences

        if idx < len(models_to_run) - 1 and switch_delay > 0:
            print(f"Waiting {switch_delay} seconds before next model...")
            time.sleep(switch_delay)

    # Reorder columns and save final output
    try:
        master_df = reorder_columns_posthoc(master_df, models_to_run, num_runs)
    except Exception as e:
        print(f"Could not reorder columns post-hoc: {e}")
    master_df.to_excel(outpath, index=False)
    analysis_end = time.time()
    analysis_duration = analysis_end - analysis_start
    print(f"\nCombined analysis complete. Results saved to {outpath}")

    # Consolidated reporting: optional aggregation + metadata append
    try:
        try:
            df_report = pd.read_excel(outpath)
        except Exception as e:
            print(f"Could not read {outpath} for reporting: {e}")
            df_report = None

        aggregated = False
        agg_summary = {}
        if aggregated_choice and df_report is not None:
            response_cols = [col for col in df_report.columns if re.match(r'^response_\d+', col.lower())]
            print(f"\nFound {len(response_cols)} response columns. Calculating aggregated consensus using mode={agg_mode}...")

            try:
                aggregated_consensus, aggregated_conf = compute_consensus_for_block(df_report, response_cols, mode=agg_mode, fuzzy_threshold=agg_fuzzy_thr)
                df_report["Aggregated_Consensus"] = aggregated_consensus
                df_report["Aggregated_Consensus_Confidence"] = aggregated_conf
            except Exception as e:
                print(f"Aggregated consensus failed: {e}")
                aggregated = False
            else:
                high_confidence = df_report[df_report['Aggregated_Consensus_Confidence'] >= 0.7]
                medium_confidence = df_report[(df_report['Aggregated_Consensus_Confidence'] >= 0.4) & (df_report['Aggregated_Consensus_Confidence'] < 0.7)]
                low_confidence = df_report[df_report['Aggregated_Consensus_Confidence'] < 0.4]
                agg_summary = {
                    'high': len(high_confidence),
                    'medium': len(medium_confidence),
                    'low': len(low_confidence),
                    'low_rows': low_confidence
                }
                aggregated = True

                try:
                    df_report.to_excel(outpath, index=False)
                    print(f"\nAggregated consensus calculation complete. Results written to {outpath}")
                except Exception as e:
                    print(f"Could not write aggregated consensus to {outpath}: {e}")

            between_model_done = False
            if between_model and len(metadata_models) > 1:
                within_model_cons_cols = [f"Consensus ({m})" for m in metadata_models if f"Consensus ({m})" in df_report.columns]
                if not within_model_cons_cols:
                    within_model_cons_cols = [c for c in df_report.columns if c.lower().startswith('consensus (')]
                if within_model_cons_cols:
                    print(f"Running between-model consensus on columns: {within_model_cons_cols}")
                    try:
                        between_cons, between_conf = compute_consensus_for_block(df_report, within_model_cons_cols, mode=between_model_mode, fuzzy_threshold=between_model_fuzzy)
                        df_report['BetweenModel_Consensus'] = between_cons
                        df_report['BetweenModel_Consensus_Confidence'] = between_conf
                        between_model_done = True
                        print("Between-model consensus computed and added to the output file.")
                    except Exception as e:
                        print(f"Between-model consensus failed: {e}")
                else:
                    print("No within-model consensus columns found for between-model aggregation.")
                if between_model_done:
                    try:
                        df_report.to_excel(outpath, index=False)
                        print(f"Between-model consensus saved to {outpath}")
                    except Exception as e:
                        print(f"Could not save between-model consensus to {outpath}: {e}")

        # Append metadata to workbook
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
                ws.append([f"Save-every interval: {save_every} rows"])
                ws.append([f"Delay between model runs: {switch_delay} seconds"])
                ws.append([f"Context mode: stateless (generate)"])
                ws.append([f"Within-model consensus enabled during runs: {within_model}"])
                ws.append([f"Within-model consensus mode: {within_model_mode}"])
                if within_model_mode == 'fuzzy':
                    ws.append([f"Within-model fuzzy threshold: {within_model_fuzzy}"])
                hours, rem = divmod(analysis_duration, 3600)
                minutes, seconds = divmod(rem, 60)
                ws.append([f"Duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s"])

                if aggregated:
                    ws.append([])
                    ws.append(["Aggregated consensus across response columns:"])
                    ws.append([f"High confidence (>=70%): {agg_summary.get('high', 0)} rows"])
                    ws.append([f"Medium confidence (40-69%): {agg_summary.get('medium', 0)} rows"])
                    ws.append([f"Low confidence (<40%): {agg_summary.get('low', 0)} rows"])
                    if agg_summary.get('low', 0) > 0:
                        ws.append(["Rows with low confidence may require manual review:"])
                        low_rows = agg_summary.get('low_rows')
                        if low_rows is not None:
                            for row_idx, row in low_rows.iterrows():
                                id_display = row.get('Identifier', row_idx + 1)
                                conf_col = 'Aggregated_Consensus_Confidence' if 'Aggregated_Consensus_Confidence' in row else 'Consensus_Confidence'
                                try:
                                    conf_val = row[conf_col]
                                    ws.append([f"Row {row_idx + 1}: {id_display} (confidence: {conf_val:.1%})"])
                                except Exception:
                                    ws.append([f"Row {row_idx + 1}: {id_display} (confidence: unknown)"])

                try:
                    import cpuinfo
                    cpu = cpuinfo.get_cpu_info()
                    cpu_brand = cpu.get('brand_raw', 'Unknown CPU')
                except Exception:
                    cpu_brand = platform.processor() or platform.machine() or 'Unknown CPU'
                ws.append([f"CPU: {cpu_brand}"])
                gpu_report = detect_gpus(cpu_brand)
                ws.append([f"GPU: {gpu_report}"])

                wb.save(outpath)
                print(f"Reporting information appended to {outpath}")
            except Exception as e:
                print(f"Could not append reporting metadata: {e}")

    except Exception as e:
        import traceback
        print(f"Reporting/aggregation error: {e}")
        traceback.print_exc()


def detect_gpus(cpu_brand):
    """
    Detects GPU information based on the current platform.
    Returns a string describing integrated and detected GPUs.
    """
    gpu_info = None
    integrated_gpu = None
    for brand in ["Radeon", "NVIDIA", "Intel Graphics", "Iris", "GeForce", "RTX", "GTX"]:
        if brand.lower() in cpu_brand.lower():
            integrated_gpu = cpu_brand
            break
    try:
        if sys.platform.startswith('win'):
            gpu_result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True)
            gpu_lines = gpu_result.stdout.splitlines()
            gpus = [line.strip() for line in gpu_lines if line.strip() and 'Name' not in line]
            gpu_info = ', '.join(gpus) if gpus else None
        elif sys.platform.startswith('linux'):
            gpu_result = subprocess.run('lspci | grep -i vga', shell=True, capture_output=True, text=True)
            gpus = [line for line in gpu_result.stdout.splitlines() if line]
            gpu_info = ', '.join(gpus) if gpus else None
            gpu_result_3d = subprocess.run('lspci | grep -i 3d', shell=True, capture_output=True, text=True)
            gpus_3d = [line for line in gpu_result_3d.stdout.splitlines() if line]
            if gpus_3d:
                gpu_info = gpu_info + ', ' + ', '.join(gpus_3d) if gpu_info else ', '.join(gpus_3d)
        elif sys.platform == 'darwin':
            gpu_result = subprocess.run(['system_profiler', 'SPDisplaysDataType', '-detailLevel', 'mini'], capture_output=True, text=True)
            gpus = [line.strip() for line in gpu_result.stdout.splitlines() if 'Chipset Model:' in line or 'Vendor:' in line]
            gpu_info = ', '.join(gpus) or None
    except Exception:
        gpu_info = None
    all_gpus = []
    if integrated_gpu:
        all_gpus.append(f"Integrated GPU: {integrated_gpu}")
    if gpu_info:
        all_gpus.append(f"Detected GPU(s): {gpu_info}")
    return ', '.join(all_gpus) if all_gpus else 'Not detected'


if __name__ == '__main__':
    main()