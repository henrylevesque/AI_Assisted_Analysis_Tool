import os
import time
import pandas as pd
from tqdm import tqdm
from ollama import chat
import sys
import re
import platform
import subprocess
from collections import Counter
import argparse
import json
try:
    import yaml
except Exception:
    yaml = None


def list_image_files(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    # return a stable, sorted list with duplicates removed
    seen = set()
    out = []
    for f in sorted(files):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def load_config(path: str):
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


def run_model_on_images(model_name, images, data_input_folder, prompt_template, num_runs):
    """Run a single model across images and return a list of dict rows with Response_{i} keys labeled by run index.

    Shows a tqdm progress bar for images and an inner runs progress bar.
    """
    rows = []
    for idx, img in enumerate(tqdm(images, desc=f"{model_name} images", unit="img", file=sys.stdout, dynamic_ncols=True), 1):
        img_path = os.path.join(data_input_folder, img)
        row_responses = []
        for run in tqdm(range(num_runs), desc="runs", unit="run", leave=False, file=sys.stdout, dynamic_ncols=True, total=num_runs):
            try:
                msg = {
                    "role": "user",
                    "content": prompt_template,
                    "images": [img_path],
                }
                response = chat(model=model_name, messages=[msg])
                cleaned = response['message']['content'].strip().replace('\r', ' ').replace('\n', ' ')
                row_responses.append(cleaned)
            except Exception as e:
                row_responses.append(f"Error: {e}")

        result = {"Image": img}
        for i, r in enumerate(row_responses, 1):
            result[f"Response_{i}"] = r
        rows.append(result)
    return rows


def _normalize_text(s: str) -> str:
    s = s or ''
    s = s.lower().strip()
    # remove surrounding punctuation
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    # collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _split_and_normalize_set(s: str):
    # split on commas and semicolons, strip, normalize tokens, return unique ordered list
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


def compute_consensus_for_block(df, response_cols, mode='exact', fuzzy_threshold=85):
    consensus = []
    confidences = []
    for _, row in df.iterrows():
        raw_responses = [str(row[c]) for c in response_cols if pd.notna(row[c])]
        responses = [r for r in (r.strip() for r in raw_responses) if r]
        if not responses:
            consensus.append('')
            confidences.append(0.0)
            continue

        if mode == 'exact':
            normalized = [_normalize_text(r) for r in responses]
            counts = Counter(normalized)
            most_common, count = counts.most_common(1)[0]
            confidence = count / len(responses)
            consensus.append(most_common)
            confidences.append(round(confidence, 3))

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
            top_group = groups_sorted[0]
            counts = Counter(top_group)
            rep, count = counts.most_common(1)[0]
            confidence = len(top_group) / len(responses)
            consensus.append(rep)
            confidences.append(round(confidence, 3))

        else:
            raise ValueError(f"Unknown consensus mode: {mode}")

    return consensus, confidences


def reorder_columns_posthoc(df, models, num_runs):
    cols = ['Image']
    for run_idx in range(1, num_runs + 1):
        for m in models:
            cols.append(f"Response_{run_idx} ({m})")
    for m in models:
        cols.append(f"Consensus ({m})")
        cols.append(f"Consensus_Confidence ({m})")
    existing = [c for c in cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    final_order = existing + remaining
    return df[final_order]


def main():
    print("Python executable:", sys.executable)

    # --- CLI / Config handling ---
    parser = argparse.ArgumentParser(description='Image analysis with multi-model comparisons and consensus')
    parser.add_argument('--config', '-c', help='Path to JSON or YAML config file')
    parser.add_argument('--models', help='Comma-separated model names to run (overrides config)')
    parser.add_argument('--input', help='Input folder path')
    parser.add_argument('--output', help='Output folder path')
    parser.add_argument('--runs', type=int, help='Number of runs per image')
    # consensus flags: mutually exclusive group; default None means "don't override config"
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--consensus', dest='consensus', action='store_true', help='Force consensus on')
    grp.add_argument('--no-consensus', dest='consensus', action='store_false', help='Force consensus off')
    parser.set_defaults(consensus=None)
    parser.add_argument('--consensus-mode', choices=['exact', 'set', 'fuzzy'], help='Consensus mode')
    parser.add_argument('--fuzzy-threshold', type=int, help='Fuzzy threshold (0-100)')
    parser.add_argument('--delay', type=float, help='Delay between model runs in seconds')
    parser.add_argument('--type-of-analysis', help='What to identify in images (objects, scene, text)')
    parser.add_argument('--no-interactive', action='store_true', help='Run non-interactively (require args/config for prompts)')
    args = parser.parse_args()

    cfg = {}
    if args.config:
        # If user supplied a config file but did not request fully non-interactive mode,
        # confirm they want to use the config. This prevents accidentally loading a YAML
        # when the user expects interactive prompts.
        use_cfg = True
        if not args.no_interactive:
            ans = input(f"Config file provided: {args.config}. Use this config and skip interactive prompts? (y/N): ").strip().lower()
            use_cfg = True if ans in ('y', 'yes') else False

        if use_cfg:
            try:
                cfg = load_config(args.config) or {}
            except Exception as e:
                print(f"Could not load config {args.config}: {e}")
                if args.no_interactive:
                    raise
                cfg = {}
        else:
            print("Ignoring provided config and running interactively.")
            cfg = {}

    def _get(key, default=None):
        val = getattr(args, key.replace('-', '_'), None)
        if val is not None:
            return val
        return cfg.get(key, default)

    # discover models
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        discovered = []
        for line in result.stdout.splitlines():
            if line.strip() and not line.lower().startswith('name'):
                discovered.append(line.split()[0])
    except Exception:
        discovered = []

    suggested = ["gemma3:12b", "llava:13b", "llama3.2-vision:11b", "qwen2.5vl:7b", "bakllava:7b"]

    print("Discovered models:", discovered or '<none>')
    print("Suggested vision models:", suggested)

    cli_models = _get('models')
    models_to_run = []
    if cli_models:
        models_to_run = [m.strip() for m in str(cli_models).split(',') if m.strip()]
    else:
        if not args.no_interactive:
            multi = input("Compare multiple models? (y/n): ").strip().lower() == 'y'
            if multi:
                print("Enter model names to compare separated by commas, or press Enter to use suggested models:")
                m_input = input().strip()
                if not m_input:
                    models_to_run = suggested
                else:
                    models_to_run = [m.strip() for m in m_input.split(',') if m.strip()]
            else:
                sel = input("Enter a single model to use (or press Enter for gemma3): ").strip()
                models_to_run = [sel or 'gemma3']
        else:
            models_to_run = suggested

    # Consensus selection
    do_consensus = _get('consensus') if _get('consensus') is not None else None
    if do_consensus is None and not args.no_interactive:
        do_consensus = input("Compute consensus after runs? (y/n) [y]: ").strip().lower()
        do_consensus = True if do_consensus in ('', 'y', 'yes') else False
    else:
        do_consensus = True if do_consensus in (True, 'y', 'yes', 'Y', '1') else False

    # Determine consensus mode: prefer CLI/config, otherwise prompt the user when they opt into consensus
    consensus_mode = _get('consensus-mode') if _get('consensus-mode') is not None else None
    fuzzy_threshold = _get('fuzzy-threshold') or 85
    if do_consensus and consensus_mode is None and not args.no_interactive:
        cm = input("Consensus mode to use for per-model consensus (exact/set/fuzzy) [exact]: ").strip().lower()
        consensus_mode = cm or 'exact'
    consensus_mode = consensus_mode or 'exact'

    if do_consensus and consensus_mode == 'fuzzy' and not _get('fuzzy-threshold') and not args.no_interactive:
        thr = input("Fuzzy threshold (0-100) [85]: ").strip()
        try:
            fuzzy_threshold = int(thr) if thr else 85
        except Exception:
            fuzzy_threshold = 85

    # Prompt / analysis instruction
    type_of_analysis = _get('type-of-analysis') or None
    if not type_of_analysis and not args.no_interactive:
        type_of_analysis = input("Enter what you want the program to identify within the image(s) (e.g., objects, scene, text): ").strip()
    type_of_analysis = type_of_analysis or 'objects'
    prompt_template = (
        "You are a design expert in a design review. You will be shown an image. Please tell me {what} concisely and only return {what}. "
        "If multiple items are present, separate them with commas. if you tell me anything other than {what}, you will not be helpful."
    ).format(what=type_of_analysis)

    # Input/output folders
    data_input_folder = _get('input') or (input("Enter the path to the image input folder: ").strip() if not args.no_interactive else None)
    data_output_folder = _get('output') or (input("Enter the path to the data output folder: ").strip() if not args.no_interactive else None)
    if not data_input_folder:
        raise FileNotFoundError("Input folder not specified.")
    if not data_output_folder:
        data_output_folder = os.getcwd()
    if not os.path.isdir(data_input_folder):
        raise FileNotFoundError("Input folder not found.")
    if not os.path.isdir(data_output_folder):
        os.makedirs(data_output_folder, exist_ok=True)

    images = list_image_files(data_input_folder)
    if not images:
        raise FileNotFoundError("No image files found in the specified input folder.")

    num_runs = _get('runs') or (int(input("Enter number of times to run analysis per image: ").strip()) if not args.no_interactive else 1)
    switch_delay = _get('delay') if _get('delay') is not None else None
    if switch_delay is None:
        if not args.no_interactive:
            delay_input = input("Enter delay between model runs in seconds (e.g., 1.0) or press Enter for 0: ").strip()
            switch_delay = float(delay_input) if delay_input else 0.0
        else:
            switch_delay = 0.0
    output_file_name = f"image_analysis_{len(images)}images_{num_runs}runs_multi.xlsx"
    output_file_path = os.path.join(data_output_folder, output_file_name)

    # Master dataframe construction: start with Image column
    master_df = pd.DataFrame({"Image": images})

    metadata_models = []
    analysis_start = time.time()

    for idx, model in enumerate(models_to_run):
        print(f"\nRunning model: {model}")
        metadata_models.append(model)
        rows = run_model_on_images(model, images, data_input_folder, prompt_template, num_runs)
        model_df = pd.DataFrame(rows)

        # rename response columns to include model name
        response_cols = [c for c in model_df.columns if c.lower().startswith('response')]
        renamed = {}
        for c in response_cols:
            newc = f"{c} ({model})"
            renamed[c] = newc
        model_df = model_df.rename(columns=renamed)

        # merge responses into master_df by Image
        master_df = master_df.merge(model_df, on='Image', how='left')

        # compute consensus for this model block if requested
        if do_consensus and response_cols:
            block_cols = [renamed[c] for c in response_cols]
            consensus, confidences = compute_consensus_for_block(master_df, block_cols, mode=consensus_mode, fuzzy_threshold=fuzzy_threshold)
            master_df[f"Consensus ({model})"] = consensus
            master_df[f"Consensus_Confidence ({model})"] = confidences

        # If not the last model, wait a short delay to allow model switching
        if idx < len(models_to_run) - 1 and switch_delay > 0:
            print(f"Waiting {switch_delay} seconds before next model...")
            time.sleep(switch_delay)

    # Reorder columns post-hoc: responses first, then consensus/confidence blocks
    try:
        master_df = reorder_columns_posthoc(master_df, models_to_run, num_runs)
    except Exception as e:
        print(f"Could not reorder columns post-hoc: {e}")

    # Save combined results
    master_df.to_excel(output_file_path, index=False)

    analysis_end = time.time()
    analysis_duration = analysis_end - analysis_start
    print(f"\nCombined analysis complete. Results saved to {output_file_path}")

    # Append metadata
    try:
        from openpyxl import load_workbook
        wb = load_workbook(output_file_path)
        ws = wb.active
        ws.append([])
        ws.append(["Prompt used:", prompt_template])
        ws.append(["Models used:", ', '.join(metadata_models)])
        ws.append([f"Runs per image: {num_runs}"])
        ws.append([f"Delay between model runs: {switch_delay} seconds"])
        ws.append([f"Consensus enabled: {do_consensus}"])
        ws.append([f"Consensus mode: {consensus_mode}"])
        if consensus_mode == 'fuzzy':
            ws.append([f"Fuzzy threshold: {fuzzy_threshold}"])
        hours, rem = divmod(analysis_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        ws.append([f"Duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s"])
        wb.save(output_file_path)
        print("Reporting metadata appended.")
    except Exception as e:
        print(f"Could not append reporting metadata: {e}")

    # Consolidated aggregated consensus and cross-model consensus (optional)
    try:
        # read the saved sheet for post-hoc aggregation
        try:
            df_report = pd.read_excel(output_file_path)
        except Exception as e:
            print(f"Could not read {output_file_path} for aggregation: {e}")
            df_report = None

        aggregated = False
        agg_summary = {}
        run_aggregation = input("Do you want to aggregate AI responses for consensus across the output file? (y/n): ").strip().lower()
        if run_aggregation == 'y' and df_report is not None:
            agg_mode = input(f"Aggregated consensus mode (exact/set/fuzzy) [{consensus_mode}]: ").strip().lower() or consensus_mode
            agg_fuzzy_thr = fuzzy_threshold
            if agg_mode == 'fuzzy':
                thr_in = input(f"Aggregated fuzzy threshold (0-100) [{fuzzy_threshold}]: ").strip()
                try:
                    agg_fuzzy_thr = int(thr_in) if thr_in else fuzzy_threshold
                except Exception:
                    agg_fuzzy_thr = fuzzy_threshold

            response_cols = [col for col in df_report.columns if col.lower().startswith('response')]
            print(f"\nFound {len(response_cols)} response columns. Calculating aggregated consensus using mode={agg_mode}...")

            try:
                aggregated_consensus, aggregated_conf = compute_consensus_for_block(df_report, response_cols, mode=agg_mode, fuzzy_threshold=agg_fuzzy_thr)
                df_report['Aggregated_Consensus'] = aggregated_consensus
                df_report['Aggregated_Consensus_Confidence'] = aggregated_conf
            except Exception as e:
                print(f"Aggregated consensus failed: {e}")
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
                    df_report.to_excel(output_file_path, index=False)
                    print(f"Aggregated consensus calculation complete. Results written to {output_file_path}")
                except Exception as e:
                    print(f"Could not write aggregated consensus to {output_file_path}: {e}")

            # Cross-model consensus when multiple models are present
            cross_model_done = False
            if len(metadata_models) > 1:
                cross_choice = input("Run cross-model consensus across per-model Consensus columns? (y/n): ").strip().lower() == 'y'
                if cross_choice:
                    per_model_cons_cols = [f"Consensus ({m})" for m in metadata_models if f"Consensus ({m})" in df_report.columns]
                    if not per_model_cons_cols:
                        per_model_cons_cols = [c for c in df_report.columns if c.lower().startswith('consensus (')]
                    if per_model_cons_cols:
                        print(f"Running cross-model consensus on columns: {per_model_cons_cols}")
                        try:
                            cross_cons, cross_conf = compute_consensus_for_block(df_report, per_model_cons_cols, mode=agg_mode, fuzzy_threshold=agg_fuzzy_thr)
                            df_report['CrossModel_Consensus'] = cross_cons
                            df_report['CrossModel_Consensus_Confidence'] = cross_conf
                            cross_model_done = True
                            print("Cross-model consensus computed and added to the output file.")
                        except Exception as e:
                            print(f"Cross-model consensus failed: {e}")
                    else:
                        print("No per-model consensus columns found for cross-model aggregation.")
                    if cross_model_done:
                        try:
                            df_report.to_excel(output_file_path, index=False)
                            print(f"Cross-model consensus saved to {output_file_path}")
                        except Exception as e:
                            print(f"Could not save cross-model consensus to {output_file_path}: {e}")

        # Append final reporting metadata (including aggregated summary if any)
        try:
            from openpyxl import load_workbook
            wb = load_workbook(output_file_path)
            ws = wb.active
            ws.append([])
            ws.append(["Prompt used:", prompt_template])
            ws.append(["Models used:", ', '.join(metadata_models)])
            ws.append([f"Runs per image: {num_runs}"])
            ws.append([f"Delay between model runs: {switch_delay} seconds"])
            ws.append([f"Consensus enabled during runs: {do_consensus}"])
            ws.append([f"Consensus mode (per-model): {consensus_mode}"])
            if consensus_mode == 'fuzzy':
                ws.append([f"Fuzzy threshold (per-model): {fuzzy_threshold}"])
            hours, rem = divmod(analysis_duration, 3600)
            minutes, seconds = divmod(rem, 60)
            ws.append([f"Duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s"])

            if aggregated:
                ws.append([])
                ws.append(["Aggregated consensus across response columns:"])
                ws.append([f"High confidence (≥70%): {agg_summary.get('high', 0)} rows"])
                ws.append([f"Medium confidence (40-69%): {agg_summary.get('medium', 0)} rows"])
                ws.append([f"Low confidence (<40%): {agg_summary.get('low', 0)} rows"])
                if agg_summary.get('low', 0) > 0:
                    ws.append(["Rows with low confidence may require manual review:"])
                    for idx, row in agg_summary.get('low_rows').iterrows():
                        id_display = row.get('Image', idx + 1)
                        ws.append([f"Row {idx + 1}: {id_display} (confidence: {row['Aggregated_Consensus_Confidence']:.1%})"])

            # CPU/GPU info
            try:
                import cpuinfo
                cpu = cpuinfo.get_cpu_info()
                cpu_brand = cpu.get('brand_raw', 'Unknown CPU')
            except Exception:
                cpu_brand = platform.processor() or platform.machine() or 'Unknown CPU'
            ws.append([f"CPU: {cpu_brand}"])

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
                    gpu_result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True)
                    gpus = []
                    for line in gpu_result.stdout.splitlines():
                        if 'Chipset Model:' in line or 'Vendor:' in line:
                            gpus.append(line.strip())
                    gpu_info = ', '.join(gpus) if gpus else None
            except Exception:
                gpu_info = None
            all_gpus = []
            if integrated_gpu:
                all_gpus.append(f"Integrated GPU: {integrated_gpu}")
            if gpu_info:
                all_gpus.append(f"Detected GPU(s): {gpu_info}")
            gpu_report = ', '.join(all_gpus) if all_gpus else 'Not detected'
            ws.append([f"GPU: {gpu_report}"])

            wb.save(output_file_path)
            print(f"Reporting and aggregation information appended to {output_file_path}")
        except Exception as e:
            print(f"Could not append consolidated reporting metadata: {e}")

    except Exception as e:
        print(f"Aggregation/reporting error: {e}")



if __name__ == '__main__':
    main()