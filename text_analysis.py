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
from ollama import chat


def list_input_file(folder):
    return next((f for f in os.listdir(folder) if f.endswith('.csv') or f.endswith('.xlsx')), None)


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
    parser.add_argument('--output', help='Output folder path')
    parser.add_argument('--id-col', help='Identifier column name')
    parser.add_argument('--content-col', help='Content column name')
    parser.add_argument('--runs', type=int, help='Number of runs per row')
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
    # (legacy per-model/cross-model parser options removed) Use within-model / between-model flags instead above

    # Append metadata tri-state
    grp_meta = parser.add_mutually_exclusive_group()
    grp_meta.add_argument('--append-metadata', dest='append_metadata', action='store_true', help='Append metadata to output workbook')
    grp_meta.add_argument('--no-append-metadata', dest='append_metadata', action='store_false', help='Do not append metadata to output workbook')
    parser.set_defaults(append_metadata=None)
    parser.add_argument('--no-interactive', action='store_true', help='Run non-interactively (require args/config for prompts)')
    args = parser.parse_args()

    cfg = {}
    if args.config:
        try:
            cfg = load_config(args.config) or {}
        except Exception as e:
            print(f"Could not load config {args.config}: {e}")
            if args.no_interactive:
                raise
            cfg = {}

    # If a config file was provided in interactive mode, confirm before silently using it
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
        # Try both hyphen and underscore forms in config
        if key in cfg:
            return cfg[key]
        key_underscore = key.replace('-', '_')
        if key_underscore in cfg:
            return cfg[key_underscore]
        return default
        return cfg.get(key, default)


    # Try to list available Ollama models to help the user choose
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

    # First ask whether to use a single model or compare multiple models
    suggested = ["gemma3:12b", "deepseek-ri:14b", "gpt-oss:20b"]
    cli_models = _get('models')
    models_to_run = []
    if cli_models:
        models_to_run = [m.strip() for m in str(cli_models).split(',') if m.strip()]
    else:
        # interactive selection unless running non-interactively
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
            # non-interactive and no CLI models: use discovered or suggested
            models_to_run = models if models else suggested

    data_in = _get('input') or (input('Data input folder [.] : ').strip() if not args.no_interactive else '.')
    data_out = _get('output') or (input('Data output folder [.] : ').strip() if not args.no_interactive else '.')
    if args.no_interactive:
        if data_in == '.':
            print("Warning: Using current directory ('.') as input folder. Specify --input to override.")
        if data_out == '.':
            print("Warning: Using current directory ('.') as output folder. Specify --output to override.")
        prompt_desc = input('Enter what you want the model or models to identify within the text: ').strip() or 'the main topic'
    prompt_desc = prompt_desc or 'the main topic'
    prompt_template = f'I am going to give you a chunk of text. Please identify {prompt_desc} used in the text. Do not tell me anything besides {prompt_desc} If you tell me anything besides {prompt_desc} you will not be helptful. The text is:'

    data_in = _get('input') or (input('Data input folder [.] : ').strip() if not args.no_interactive else '.')
    data_out = _get('output') or (input('Data output folder [.] : ').strip() if not args.no_interactive else '.')

    f = list_input_file(data_in)
    if not f:
        print('No input CSV/XLSX found in', data_in)
        return
    path = os.path.join(data_in, f)
    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)

    # columns and runs: attempt to pull from CLI/config, otherwise interactive
    id_col = _get('id_col') or None
    content_col = _get('content_col') or None
    num_runs = _get('runs') or None
    if not (id_col and content_col and num_runs):
        if args.no_interactive:
            # require id/content/runs in config/args
            id_col = id_col or cfg.get('id_col')
            content_col = content_col or cfg.get('content_col')
            num_runs = num_runs or cfg.get('runs') or 1
        else:
            id_col, content_col, num_runs = get_user_columns(df)
    if id_col and id_col in df.columns:
        ids = df[id_col].tolist()
    else:
        ids = list(range(1, len(df) + 1))
    if content_col not in df.columns:
        print('Content column not found. Exiting.')
        return

    contents = df[content_col].tolist()

  
    # Pre-run consensus and metadata choices (collect before long runs so non-interactive runs can proceed)
    # Within-model consensus decision (tri-state via CLI/config). Default = True.
    within_model = _get('within_model_consensus') if _get('within_model_consensus') is not None else None
    if within_model is None:
        if args.no_interactive:
            within_model = True
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

    # Aggregated (across response columns) decision
    aggregated_choice = _get('aggregate') if _get('aggregate') is not None else None
    if aggregated_choice is None:
        if args.no_interactive:
            aggregated_choice = False
        else:
            agg_resp = input("Do you want to aggregate AI responses for consensus across the output file? (y/n) [n]: ").strip().lower()
            aggregated_choice = True if agg_resp in ('y', 'yes') else False
    else:
        aggregated_choice = True if aggregated_choice in (True, 'y', 'yes', 'Y', '1') else False

    # Between-model consensus decision (tri-state). Default = True.
    between_model = _get('between_model_consensus') if _get('between_model_consensus') is not None else None
    if between_model is None:
        if args.no_interactive:
            between_model = True
        else:
            bm_resp = input("Compute between-model consensus across per-model Consensus columns? (y/n) [y]: ").strip().lower()
            between_model = True if bm_resp in ('', 'y', 'yes') else False
    else:
        between_model = True if between_model in (True, 'y', 'yes', 'Y', '1') else False

    between_model_mode = _get('between-model-consensus-mode') or within_model_mode
    between_model_fuzzy = _get('between-model-fuzzy-threshold') or within_model_fuzzy

    # Append metadata to the workbook (default True)
    append_metadata = _get('append_metadata') if _get('append_metadata') is not None else None
    if append_metadata is None:
        append_metadata = True
    else:
        append_metadata = True if append_metadata in (True, 'y', 'yes', 'Y', '1') else False

    # delay between model runs (useful if switching models manually)
    switch_delay = _get('delay') if _get('delay') is not None else 0.0
    if switch_delay is None:
        switch_delay = 0.0
    if len(models_to_run) > 1 and switch_delay == 0.0 and not args.no_interactive:
        delay_input = input("Enter delay between model runs in seconds (e.g., 1.0) or press Enter for 0: ").strip()
        switch_delay = float(delay_input) if delay_input else 0.0

    # Master dataframe: preserve Identifier and Content
    master_df = pd.DataFrame({ 'Identifier': ids, 'Content': contents })
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

    def run_model_on_texts(model_name, texts, prompt_template, num_runs, ids):
        rows = []
        for idx, txt in enumerate(tqdm(texts, desc=f"{model_name} texts", unit="row", file=sys.stdout, dynamic_ncols=True), 1):
            row_responses = []
            for run in tqdm(range(num_runs), desc="runs", unit="run", leave=False, file=sys.stdout, total=num_runs, dynamic_ncols=True):
                try:
                    resp = chat(model=model_name, messages=[{"role": "user", "content": f"{prompt_template} {txt}"}])
                    cleaned = resp['message']['content'].strip().replace('\r', ' ').replace('\n', ' ')
                    row_responses.append(cleaned)
                except Exception as e:
                    row_responses.append(f"Error: {e}")
            result = { 'Identifier': ids[idx-1] }
            for i, r in enumerate(row_responses, 1):
                result[f"Response_{i}"] = r
            rows.append(result)
        return rows

    # Run models sequentially
    for idx, m in enumerate(models_to_run):
        model = m
        print(f"\nRunning model: {model}")
        metadata_models.append(model)
        rows = run_model_on_texts(model, contents, prompt_template, num_runs, ids)
        model_df = pd.DataFrame(rows)

        # Defensive: drop duplicate Identifier rows from model output before merging to avoid merge multiplication
        if 'Identifier' in model_df.columns:
            model_df = model_df.drop_duplicates(subset=['Identifier'])

        # rename response columns to include model name
        response_cols = [c for c in model_df.columns if c.lower().startswith('response')]
        renamed = {c: f"{c} ({model})" for c in response_cols}
        model_df = model_df.rename(columns=renamed)

        # merge responses into master_df by Identifier
        master_df = master_df.merge(model_df, on='Identifier', how='left')

        # compute consensus for this model block if requested (use within-model pre-run choice)
        if within_model and response_cols:
            block_cols = [renamed[c] for c in response_cols]
            consensus, confidences = compute_consensus_for_block(master_df, block_cols, mode=within_model_mode, fuzzy_threshold=within_model_fuzzy)
            master_df[f"Consensus ({model})"] = consensus
            master_df[f"Consensus_Confidence ({model})"] = confidences

        # If not the last model, wait a short delay to allow model switching
        if idx < len(models_to_run) - 1 and switch_delay > 0:
            print(f"Waiting {switch_delay} seconds before next model...")
            time.sleep(switch_delay)

    # After running models, reorder columns and save
    base = os.path.splitext(f)[0]
    outname = f"{base}_text_analysis_{num_runs}runs_multi.xlsx" if len(models_to_run) > 1 else f"{base}_text_analysis_{num_runs}runs.xlsx"
    outpath = os.path.join(data_out, outname)
    try:
        master_df = reorder_columns_posthoc(master_df, models_to_run, num_runs)
    except Exception as e:
        print(f"Could not reorder columns post-hoc: {e}")
    master_df.to_excel(outpath, index=False)
    analysis_end = time.time()
    analysis_duration = analysis_end - analysis_start
    print(f"\nCombined analysis complete. Results saved to {outpath}")

    # Consolidated reporting: optional aggregation + metadata append using pre-run choices
    try:
        # Load dataframe from the saved output file so aggregation and reporting operate on the final sheet
        try:
            df_report = pd.read_excel(outpath)
        except Exception as e:
            print(f"Could not read {outpath} for reporting: {e}")
            df_report = None

        # Optionally run an overall aggregation across response columns (post-hoc)
        aggregated = False
        agg_summary = {}
        if aggregated_choice and df_report is not None:
            # Determine aggregated mode (default to within-model mode)
            agg_mode = within_model_mode
            if not args.no_interactive:
                agg_mode = input(f"Aggregated consensus mode (exact/set/fuzzy) [{agg_mode}]: ").strip().lower() or agg_mode
            agg_fuzzy_thr = within_model_fuzzy
            if agg_mode == 'fuzzy' and not args.no_interactive:
                thr_in = input(f"Aggregated fuzzy threshold (0-100) [{agg_fuzzy_thr}]: ").strip()
                try:
                    agg_fuzzy_thr = int(thr_in) if thr_in else agg_fuzzy_thr
                except Exception:
                    pass

            response_cols = [col for col in df_report.columns if re.match(r'^response_\d+', col.lower())]
            print(f"\nFound {len(response_cols)} response columns. Calculating aggregated consensus using mode={agg_mode}...")

            # Use compute_consensus_for_block (supports exact/set/fuzzy)
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

                # Save back the consensus columns to the output file
                try:
                    df_report.to_excel(outpath, index=False)
                    print(f"\nAggregated consensus calculation complete. Results written to {outpath}")
                except Exception as e:
                    print(f"Could not write aggregated consensus to {outpath}: {e}")

            # If multiple models were used, optionally compute between-model consensus across within-model Consensus columns
            between_model_done = False
            if between_model and len(metadata_models) > 1:
                # build within-model consensus column list (only include those present)
                within_model_cons_cols = [f"Consensus ({m})" for m in metadata_models if f"Consensus ({m})" in df_report.columns]
                if not within_model_cons_cols:
                    # fallback: try any column that starts with 'consensus ('
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
                # save after between-model consensus if added
                if between_model_done:
                    try:
                        df_report.to_excel(outpath, index=False)
                        print(f"Between-model consensus saved to {outpath}")
                    except Exception as e:
                        print(f"Could not save between-model consensus to {outpath}: {e}")

        # Build reporting metadata and append to the bottom of the workbook
        try:
            from openpyxl import load_workbook
            wb = load_workbook(outpath)
            ws = wb.active
            ws.append([])
            ws.append(["Prompt used:", prompt_template])
            ws.append(["Models used:", ', '.join(metadata_models)])
            ws.append([f"Runs per row: {num_runs}"])
            ws.append([f"Delay between model runs: {switch_delay} seconds"])
            ws.append([f"Within-model consensus enabled during runs: {within_model}"])
            ws.append([f"Within-model consensus mode: {within_model_mode}"])
            if within_model_mode == 'fuzzy':
                ws.append([f"Within-model fuzzy threshold: {within_model_fuzzy}"])
            hours, rem = divmod(analysis_duration, 3600)
            minutes, seconds = divmod(rem, 60)
            ws.append([f"Duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s"])

            # If we ran the aggregated consensus, add that summary
            if aggregated:
                ws.append([])
                ws.append(["Aggregated consensus across response columns:"])
                ws.append([f"High confidence (≥70%): {agg_summary.get('high', 0)} rows"])
                ws.append([f"Medium confidence (40-69%): {agg_summary.get('medium', 0)} rows"])
                ws.append([f"Low confidence (<40%): {agg_summary.get('low', 0)} rows"])
                if agg_summary.get('low', 0) > 0:
                    ws.append(["Rows with low confidence may require manual review:"])
                    for idx, row in agg_summary.get('low_rows').iterrows():
                        id_display = row.get('Identifier', idx + 1)
                        conf_col = 'Aggregated_Consensus_Confidence' if 'Aggregated_Consensus_Confidence' in row else 'Consensus_Confidence'
                        ws.append([f"Row {idx + 1}: {id_display} (confidence: {row[conf_col]:.1%})"])

            # CPU/GPU info
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
            print(f"Could not append reporting metadata (while adding reporting information to the Excel file): {e}")

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
    # Check for integrated GPU by CPU brand
    for brand in ["Radeon", "NVIDIA", "Intel Graphics", "Iris", "GeForce", "RTX", "GTX"]:
        if brand.lower() in cpu_brand.lower():
            integrated_gpu = cpu_brand
            break
    try:
        if sys.platform.startswith('win'):
            # Windows: Use WMIC to get GPU names
            gpu_result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True)
            gpu_lines = gpu_result.stdout.splitlines()
            gpus = [line.strip() for line in gpu_lines if line.strip() and 'Name' not in line]
            gpu_info = ', '.join(gpus) if gpus else None
        elif sys.platform.startswith('linux'):
            # Linux: Use lspci to get VGA and 3D controller info
            gpu_result = subprocess.run('lspci | grep -i vga', shell=True, capture_output=True, text=True)
            gpus = [line for line in gpu_result.stdout.splitlines() if line]
            gpu_info = ', '.join(gpus) if gpus else None
            gpu_result_3d = subprocess.run('lspci | grep -i 3d', shell=True, capture_output=True, text=True)
            gpus_3d = [line for line in gpu_result_3d.stdout.splitlines() if line]
            if gpus_3d:
                gpu_info = gpu_info + ', ' + ', '.join(gpus_3d) if gpu_info else ', '.join(gpus_3d)
        elif sys.platform == 'darwin':
            # macOS: Use system_profiler to get GPU info
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