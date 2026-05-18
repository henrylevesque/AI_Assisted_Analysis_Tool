import os
import time
import pandas as pd
from tqdm import tqdm
from ollama import generate
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
    """
    List image files in the given folder with supported extensions.
    Returns a stable, sorted list of unique image filenames.
    """
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]
    seen = set()
    out = []
    for f in sorted(files):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _clean_path_helper(p: str):
    """Helper to clean path (remove quotes, whitespace)."""
    if p is None:
        return p
    p = str(p).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return p


def is_image_file(path):
    """Check if path is an image file."""
    if not path:
        return False
    p = _clean_path_helper(path)
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    return os.path.isfile(p) and p.lower().endswith(exts)


def resolve_input_images(input_path) -> tuple:
    """
    Resolve input to images (single file or folder of images).

    Args:
        input_path: Path to an image file or folder of images

    Returns:
        (images_list, input_folder) where images_list is filenames and input_folder is the directory

    Raises:
        FileNotFoundError: If path cannot be found or no valid images found
    """
    if not input_path:
        raise ValueError("Input path is required")

    input_path = _clean_path_helper(input_path)

    if is_image_file(input_path):
        folder = os.path.dirname(input_path) or '.'
        filename = os.path.basename(input_path)
        return [filename], folder

    if os.path.isdir(input_path):
        images = list_image_files(input_path)
        if not images:
            raise FileNotFoundError(f"No image files found in folder: {input_path}")
        return images, input_path

    raise FileNotFoundError(f"Input path not found: {input_path}")


def resolve_output_path(input_path, output_spec, num_images, num_runs, num_models) -> str:
    """
    Resolve output file path from various output specifications.
    """
    input_path = _clean_path_helper(input_path)
    output_spec = _clean_path_helper(output_spec) if output_spec else None

    if not output_spec:
        if os.path.isfile(input_path):
            output_dir = os.path.dirname(input_path) or '.'
        else:
            output_dir = input_path
    elif os.path.isdir(output_spec):
        output_dir = output_spec
    elif output_spec.endswith('.xlsx'):
        return output_spec
    else:
        output_dir = output_spec
        os.makedirs(output_dir, exist_ok=True)

    suffix = f"{num_images}images_{num_runs}runs_multi.xlsx" if num_models > 1 else f"{num_images}images_{num_runs}runs.xlsx"
    output_filename = f"image_analysis_{suffix}"

    return os.path.join(output_dir, output_filename)


def load_config(path):
    """
    Load a JSON or YAML configuration file from path and return the parsed dict.
    """
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


def run_model_on_images(model_name, images, data_input_folder, prompt_template, num_runs, output_file_path, save_every=10):
    """
    Run a single model across images using stateless generate calls (no accumulated context).
    Saves interim results every save_every images to protect against crashes.

    Note: Images are passed directly to generate() rather than via chat messages,
    which prevents context accumulation across runs and images.
    """
    rows = []
    for idx, img in enumerate(tqdm(images, desc=f"{model_name} images", unit="img", file=sys.stdout, dynamic_ncols=True), 1):
        img_path = os.path.join(data_input_folder, img)
        row_responses = []
        for run in tqdm(range(num_runs), desc="runs", unit="run", leave=False, file=sys.stdout, dynamic_ncols=True, total=num_runs):
            try:
                # Use generate (stateless) with images at top level — prevents context accumulation
                resp = generate(model=model_name, prompt=prompt_template, images=[img_path])
                cleaned = resp['response'].strip().replace('\r', ' ').replace('\n', ' ')
                row_responses.append(cleaned)
            except Exception as e:
                row_responses.append(f"Error: {e}")

        result = {"Image": img}
        for i, r in enumerate(row_responses, 1):
            result[f"Response_{i}"] = r
        rows.append(result)

        # Incremental save every save_every images
        if idx % save_every == 0:
            try:
                interim_df = pd.DataFrame(rows)
                interim_path = output_file_path.replace('.xlsx', f'_interim_{idx}images.xlsx')
                interim_df.to_excel(interim_path, index=False)
                print(f"\n  [Checkpoint] Saved {idx} images to {interim_path}")
            except Exception as e:
                print(f"\n  [Checkpoint] Could not save interim at image {idx}: {e}")

    return rows


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

    parser = argparse.ArgumentParser(description='Image analysis with multi-model comparisons and consensus')
    parser.add_argument('--config', '-c', help='Path to JSON or YAML config file')
    parser.add_argument('--models', help='Comma-separated model names to run (overrides config)')
    parser.add_argument('--input', help='Input image file or folder path')
    parser.add_argument('--output', help='Output folder path')
    parser.add_argument('--runs', type=int, help='Number of runs per image')
    parser.add_argument('--save-every', type=int, default=10, help='Save interim results every N images (default: 10)')
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
    parser.add_argument('--delay', type=float)
    parser.add_argument('--type-of-analysis', help='What to identify in images (e.g., objects, scene, text)')
    parser.add_argument('--aggregate', dest='aggregate', action='store_true')
    parser.add_argument('--no-aggregate', dest='aggregate', action='store_false')
    parser.set_defaults(aggregate=None)
    parser.add_argument('--append-metadata', dest='append_metadata', action='store_true')
    parser.add_argument('--no-append-metadata', dest='append_metadata', action='store_false')
    parser.set_defaults(append_metadata=None)
    parser.add_argument('--no-interactive', action='store_true')
    args = parser.parse_args()

    cfg = {}
    if args.config:
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
        if key in cfg:
            return cfg.get(key)
        alt = key.replace('-', '_')
        if alt in cfg:
            return cfg.get(alt)
        return default

    # Discover available models
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        discovered = []
        for line in result.stdout.splitlines():
            if line.strip() and not line.lower().startswith('name'):
                discovered.append(line.split()[0])
    except Exception:
        discovered = []

    suggested = ["gemma3:12b", "llava:13b", "llama3.2-vision:11b"]
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
                models_to_run = [m.strip() for m in m_input.split(',') if m.strip()] if m_input else suggested
            else:
                sel = input("Enter a single model to use (or press Enter for gemma3:12b): ").strip()
                models_to_run = [sel or 'gemma3:12b']
        else:
            models_to_run = suggested

    # Within-model consensus
    within_model = _get('within-model-consensus')
    if within_model is None:
        if not args.no_interactive:
            ans = input("Compute within-model consensus after runs? (y/n) [y]: ").strip().lower()
            within_model = True if ans in ('', 'y', 'yes') else False
        else:
            within_model = True
    else:
        within_model = True if within_model in (True, 'y', 'yes', 'Y', '1') else False

    within_model_mode = _get('within-model-consensus-mode')
    within_model_fuzzy_threshold = _get('within-model-fuzzy-threshold') or 85
    if within_model and within_model_mode is None and not args.no_interactive:
        cm = input("Within-model consensus mode (exact/set/fuzzy) [exact]: ").strip().lower()
        within_model_mode = cm or 'exact'
    within_model_mode = within_model_mode or 'exact'

    if within_model and within_model_mode == 'fuzzy' and not _get('within-model-fuzzy-threshold') and not args.no_interactive:
        thr = input("Within-model fuzzy threshold (0-100) [85]: ").strip()
        try:
            within_model_fuzzy_threshold = int(thr) if thr else 85
        except Exception:
            within_model_fuzzy_threshold = 85

    # Between-model consensus
    between_model = _get('between-model-consensus')
    between_model_mode = _get('between-model-consensus-mode')
    between_model_fuzzy_threshold = _get('between-model-fuzzy-threshold') or within_model_fuzzy_threshold

    if between_model is None:
        if not args.no_interactive:
            ans = input("Compute between-model consensus after runs? (y/n) [y]: ").strip().lower()
            between_model = True if ans in ('', 'y', 'yes') else False
        else:
            between_model = True
    else:
        between_model = True if between_model in (True, 'y', 'yes', 'Y', '1') else False

    between_model_mode = between_model_mode or within_model_mode
    if between_model and not args.no_interactive:
        bm_mode = input(f"Between-model consensus mode (exact/set/fuzzy) [{between_model_mode}]: ").strip().lower()
        between_model_mode = bm_mode or between_model_mode
    if between_model and between_model_mode == 'fuzzy' and not _get('between-model-fuzzy-threshold') and not args.no_interactive:
        thr_b = input(f"Between-model fuzzy threshold (0-100) [{between_model_fuzzy_threshold}]: ").strip()
        try:
            between_model_fuzzy_threshold = int(thr_b) if thr_b else between_model_fuzzy_threshold
        except Exception:
            pass

    # Aggregation
    aggregated_choice = _get('aggregate')
    if aggregated_choice is None:
        if args.no_interactive:
            aggregated_choice = False
        else:
            agg_resp = input("Do you want to aggregate AI responses for consensus across the output file? (y/n) [n]: ").strip().lower()
            aggregated_choice = True if agg_resp in ('y', 'yes') else False
    else:
        aggregated_choice = True if aggregated_choice in (True, 'y', 'yes', 'Y', '1') else False

    agg_mode = within_model_mode
    agg_fuzzy_thr = within_model_fuzzy_threshold
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

    # Metadata
    append_metadata = _get('append-metadata')
    if append_metadata is None and not args.no_interactive:
        ans = input("Append metadata/reporting to the output Excel file? (Y/n) [Y]: ").strip().lower()
        append_metadata = True if ans in ('', 'y', 'yes') else False
    else:
        append_metadata = True if append_metadata in (True, 'y', 'yes', '1') else False

    # Prompt
    type_of_analysis = _get('type-of-analysis') or None
    if not type_of_analysis and not args.no_interactive:
        type_of_analysis = input("Enter what you want the program to identify within the image(s) (e.g., objects, scene, text): ").strip()
    type_of_analysis = type_of_analysis or 'objects'
    prompt_template = (
        "You are a design expert in a design review. You will be shown an image. Please tell me {what} concisely and only return {what}. "
        "If multiple items are present, separate them with commas. If you tell me anything other than {what}, you will not be helpful."
    ).format(what=type_of_analysis)

    # Input/output
    data_input_folder = _get('input') or (input("Enter the path to an image file or folder: ").strip() if not args.no_interactive else None)
    data_output_folder = _get('output') or (input("Enter the path to the data output file or folder [same as input]: ").strip() if not args.no_interactive else None)
    if not data_input_folder:
        raise FileNotFoundError("Input path not specified.")

    try:
        images, input_folder = resolve_input_images(data_input_folder)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: {e}")

    if not images:
        raise FileNotFoundError("No image files found.")

    num_runs = _get('runs') or (int(input("Enter number of times to run analysis per image: ").strip()) if not args.no_interactive else 1)
    save_every = _get('save_every') or args.save_every or 10

    switch_delay = _get('delay')
    if switch_delay is None:
        if not args.no_interactive:
            delay_input = input("Enter delay between model runs in seconds (e.g., 1.0) or press Enter for 0: ").strip()
            switch_delay = float(delay_input) if delay_input else 0.0
        else:
            switch_delay = 0.0

    output_file_path = resolve_output_path(data_input_folder, data_output_folder, len(images), num_runs, len(models_to_run))

    output_dir = os.path.dirname(output_file_path) or '.'
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved to: {output_file_path}")
    except Exception as e:
        print(f"Warning: could not create output directory {output_dir}: {e}")

    master_df = pd.DataFrame({"Image": images})
    metadata_models = []
    analysis_start = time.time()

    for idx, model in enumerate(models_to_run):
        print(f"\nRunning model: {model}")
        metadata_models.append(model)
        rows = run_model_on_images(model, images, input_folder, prompt_template, num_runs, output_file_path, save_every=save_every)
        model_df = pd.DataFrame(rows)

        response_cols = [c for c in model_df.columns if c.lower().startswith('response')]
        renamed = {c: f"{c} ({model})" for c in response_cols}
        model_df = model_df.rename(columns=renamed)

        master_df = master_df.merge(model_df, on='Image', how='left')

        if within_model and response_cols:
            block_cols = [renamed[c] for c in response_cols]
            consensus, confidences = compute_consensus_for_block(master_df, block_cols, mode=within_model_mode, fuzzy_threshold=within_model_fuzzy_threshold)
            master_df[f"Consensus ({model})"] = consensus
            master_df[f"Consensus_Confidence ({model})"] = confidences

        if idx < len(models_to_run) - 1 and switch_delay > 0:
            print(f"Waiting {switch_delay} seconds before next model...")
            time.sleep(switch_delay)

    try:
        master_df = reorder_columns_posthoc(master_df, models_to_run, num_runs)
    except Exception as e:
        print(f"Could not reorder columns post-hoc: {e}")

    master_df.to_excel(output_file_path, index=False)

    analysis_end = time.time()
    analysis_duration = analysis_end - analysis_start
    print(f"\nCombined analysis complete. Results saved to {output_file_path}")

    # Aggregation and reporting
    try:
        try:
            df_report = pd.read_excel(output_file_path)
        except Exception as e:
            print(f"Could not read {output_file_path} for aggregation: {e}")
            df_report = None

        aggregated = False
        agg_summary = {}

        if aggregated_choice and df_report is not None:
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
                    print(f"Aggregated consensus complete. Results written to {output_file_path}")
                except Exception as e:
                    print(f"Could not write aggregated consensus: {e}")

            between_model_done = False
            if between_model and len(metadata_models) > 1:
                within_model_cons_cols = [f"Consensus ({m})" for m in metadata_models if f"Consensus ({m})" in df_report.columns]
                if not within_model_cons_cols:
                    within_model_cons_cols = [c for c in df_report.columns if c.lower().startswith('consensus (')]
                if within_model_cons_cols:
                    print(f"Running between-model consensus on columns: {within_model_cons_cols}")
                    try:
                        between_cons, between_conf = compute_consensus_for_block(df_report, within_model_cons_cols, mode=agg_mode, fuzzy_threshold=agg_fuzzy_thr)
                        df_report['BetweenModel_Consensus'] = between_cons
                        df_report['BetweenModel_Consensus_Confidence'] = between_conf
                        between_model_done = True
                        print("Between-model consensus computed.")
                    except Exception as e:
                        print(f"Between-model consensus failed: {e}")
                if between_model_done:
                    try:
                        df_report.to_excel(output_file_path, index=False)
                    except Exception as e:
                        print(f"Could not save between-model consensus: {e}")

        if append_metadata:
            try:
                from openpyxl import load_workbook
                wb = load_workbook(output_file_path)
                ws = wb.active
                if ws is None:
                    raise ValueError("Could not access worksheet")
                ws.append([])
                ws.append(["Prompt used:", prompt_template])
                ws.append(["Models used:", ', '.join(metadata_models)])
                ws.append([f"Runs per image: {num_runs}"])
                ws.append([f"Save-every interval: {save_every} images"])
                ws.append([f"Delay between model runs: {switch_delay} seconds"])
                ws.append([f"Context mode: stateless (generate)"])
                ws.append([f"Within-model consensus enabled: {within_model}"])
                ws.append([f"Within-model consensus mode: {within_model_mode}"])
                if within_model_mode == 'fuzzy':
                    ws.append([f"Within-model fuzzy threshold: {within_model_fuzzy_threshold}"])
                if len(metadata_models) > 1:
                    ws.append([f"Between-model consensus enabled: {between_model}"])
                    ws.append([f"Between-model consensus mode: {between_model_mode}"])
                    if between_model_mode == 'fuzzy':
                        ws.append([f"Between-model fuzzy threshold: {between_model_fuzzy_threshold}"])
                ws.append([f"Aggregated consensus enabled: {aggregated_choice}"])
                if aggregated_choice:
                    ws.append([f"Aggregated consensus mode: {agg_mode}"])
                    if agg_mode == 'fuzzy':
                        ws.append([f"Aggregated fuzzy threshold: {agg_fuzzy_thr}"])
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
                                id_display = row.get('Image', row_idx + 1)
                                try:
                                    conf_val = row['Aggregated_Consensus_Confidence']
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
                        gpus = [line.strip() for line in gpu_result.stdout.splitlines() if 'Chipset Model:' in line or 'Vendor:' in line]
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
                print(f"Reporting information appended to {output_file_path}")
            except Exception as e:
                print(f"Could not append reporting metadata: {e}")

    except Exception as e:
        print(f"Aggregation/reporting error: {e}")


if __name__ == '__main__':
    main()