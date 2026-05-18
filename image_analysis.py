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
    'models', 'input', 'output', 'type_of_analysis',
    'runs', 'timeout', 'retries', 'delay',
    'within_model_consensus', 'within_model_consensus_mode', 'within_model_fuzzy_threshold',
    'between_model_consensus', 'between_model_consensus_mode', 'between_model_fuzzy_threshold',
    'aggregate', 'aggregated_consensus_mode', 'aggregate_consensus_mode',
    'aggregated_fuzzy_threshold', 'aggregate_fuzzy_threshold',
    'append_metadata',
}


def list_image_files(folder: str) -> list[str]:
    """Return a stable sorted list of unique image filenames in folder."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]
    seen: set[str] = set()
    out: list[str] = []
    for f in sorted(files):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _clean_path_helper(p: Optional[str]) -> Optional[str]:
    """Strip surrounding quotes and whitespace from a path string."""
    if p is None:
        return p
    p = str(p).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return p


def is_image_file(path: str) -> bool:
    """Return True if path points to a supported image file."""
    if not path:
        return False
    p = _clean_path_helper(path)
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    return os.path.isfile(p) and p.lower().endswith(exts)


def resolve_input_images(input_path: str) -> tuple[list[str], str]:
    """
    Resolve input to a list of image filenames and their parent folder.

    Returns (filenames, folder). Raises FileNotFoundError if nothing found.
    """
    if not input_path:
        raise ValueError("Input path is required")

    input_path = _clean_path_helper(input_path)

    if is_image_file(input_path):
        return [os.path.basename(input_path)], os.path.dirname(os.path.abspath(input_path))

    if os.path.isdir(input_path):
        images = list_image_files(input_path)
        if not images:
            raise FileNotFoundError(f"No image files found in folder: {input_path}")
        return images, input_path

    raise FileNotFoundError(f"Input path not found: {input_path}")


def resolve_output_path(input_folder: str, output_spec: Optional[str], num_images: int, num_runs: int, num_models: int) -> str:
    """
    Resolve the output Excel file path from user-supplied output spec.

    Falls back to the input folder with an auto-generated name.
    Appends a timestamp if the resolved path already exists.
    """
    output_spec = _clean_path_helper(output_spec) if output_spec else None

    def _add_ts_if_exists(path: str) -> str:
        if os.path.exists(path):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(path)
            return f"{base}_{ts}{ext}"
        return path

    if not output_spec:
        output_dir = input_folder
    elif os.path.isdir(output_spec):
        output_dir = output_spec
    elif output_spec.endswith('.xlsx'):
        return _add_ts_if_exists(output_spec)
    else:
        output_dir = output_spec
        os.makedirs(output_dir, exist_ok=True)

    suffix = f"{num_images}images_{num_runs}runs_multi.xlsx" if num_models > 1 else f"{num_images}images_{num_runs}runs.xlsx"
    return _add_ts_if_exists(os.path.join(output_dir, f"image_analysis_{suffix}"))


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
    df: pd.DataFrame,
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
    for _, row in df.iterrows():
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


def reorder_columns_posthoc(df: pd.DataFrame, models: list[str], num_runs: int) -> pd.DataFrame:
    """Reorder output columns: Image, responses by run, consensus blocks, cross-model columns."""
    cols = ['Image']
    for run_idx in range(1, num_runs + 1):
        for m in models:
            cols.append(f"Response_{run_idx} ({m})")
    for m in models:
        cols.append(f"Consensus ({m})")
        cols.append(f"Confidence ({m})")
    cols += ['Between_Consensus', 'Between_Confidence', 'Aggregate_Consensus', 'Aggregate_Confidence']
    existing = [c for c in cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df[existing + remaining]


def _ollama_call(
    client: ollama.Client,
    model: str,
    prompt: str,
    images: Optional[list[str]] = None,
    max_retries: int = 2,
) -> str:
    """Call ollama.generate with exponential-backoff retry. Returns cleaned response text."""
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            kwargs: dict = {'model': model, 'prompt': prompt}
            if images:
                kwargs['images'] = images
            resp = client.generate(**kwargs)
            return resp['response'].strip().replace('\r', ' ').replace('\n', ' ')
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(2.0 * (2 ** attempt))
    return f"Error: {last_error}"


def run_model_on_images(
    client: ollama.Client,
    model_name: str,
    images: list[str],
    input_folder: str,
    prompt_template: str,
    num_runs: int,
    max_retries: int = 2,
) -> list[dict]:
    """Run a single model across all images and return a list of response dicts."""
    rows = []
    for img in tqdm(images, desc=f"{model_name} images", unit="img", file=sys.stdout, dynamic_ncols=True):
        img_path = os.path.join(input_folder, img)
        row_responses = []
        for _ in tqdm(range(num_runs), desc="runs", unit="run", leave=False, file=sys.stdout, dynamic_ncols=True, total=num_runs):
            result = _ollama_call(client, model_name, prompt_template, images=[img_path], max_retries=max_retries)
            row_responses.append(result)
        row: dict = {'Image': img}
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
    print("Python executable:", sys.executable)

    parser = argparse.ArgumentParser(description='Image analysis with multi-model comparisons and consensus')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--config', '-c', help='Path to JSON or YAML config file')
    parser.add_argument('--models', help='Comma-separated model names to run (overrides config)')
    parser.add_argument('--input', help='Input image file or folder path')
    parser.add_argument('--output', help='Output file or folder path')
    parser.add_argument('--type-of-analysis', help='What to identify in the images (e.g., "objects", "the land use type")')
    parser.add_argument('--runs', type=int, help='Number of runs per image')
    parser.add_argument('--timeout', type=float, help='Ollama request timeout in seconds [120]')
    parser.add_argument('--retries', type=int, help='Number of retries on Ollama failure [2]')
    parser.add_argument('--delay', type=float, help='Delay between model runs in seconds')

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
        use_cfg = True
        if not args.no_interactive:
            ans = input(f"Load configuration from {args.config}? (y/N): ").strip().lower()
            use_cfg = ans in ('y', 'yes')
        if use_cfg:
            try:
                cfg = load_config(args.config) or {}
            except Exception as e:
                print(f"Could not load config {args.config}: {e}")
                if args.no_interactive:
                    raise
                cfg = {}
            unknown = set(cfg.keys()) - KNOWN_CONFIG_KEYS
            if unknown:
                print(f"Warning: unknown config key(s): {', '.join(sorted(unknown))}. These will be ignored.")
        else:
            print("Ignoring config, running interactively.")

    def _get(key: str, default=None):
        val = getattr(args, key.replace('-', '_'), None)
        if val is not None:
            return val
        if key in cfg:
            return cfg[key]
        alt = key.replace('-', '_')
        if alt in cfg:
            return cfg[alt]
        return default

    # --- Model selection ---
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        discovered = [l.split()[0] for l in result.stdout.splitlines() if l.strip() and not l.lower().startswith('name')]
    except Exception:
        discovered = []

    suggested = ["llava:13b", "llama3.2-vision:11b", "qwen2.5vl:7b", "gemma3:12b"]
    if discovered:
        print("Available Ollama models:")
        for i, m in enumerate(discovered, 1):
            print(f"  {i}. {m}")
    else:
        print("Suggested vision models:", ', '.join(suggested))

    cli_models = _get('models')
    models_to_run: list[str] = []
    if cli_models:
        models_to_run = [m.strip() for m in str(cli_models).split(',') if m.strip()]
    elif not args.no_interactive:
        multi = input("\nDo you want to run multiple models and compare results? (y/n): ").strip().lower() == 'y'
        if multi:
            print("Enter model names separated by commas (or press Enter to use suggested vision models):")
            m_input = input().strip()
            models_to_run = [m.strip() for m in m_input.split(',') if m.strip()] if m_input else suggested
        else:
            sel = input('Enter a model name (or press Enter for llava:13b): ').strip()
            models_to_run = [sel or 'llava:13b']
    else:
        models_to_run = suggested

    # --- Input / output paths ---
    data_input = _get('input') or (input('\nPath to input image file or folder: ').strip() if not args.no_interactive else None)
    if not data_input:
        raise FileNotFoundError("Input path not specified.")

    data_output = _get('output') or (input('Path to output file or folder (Enter to save alongside input images): ').strip() if not args.no_interactive else None)

    # Resolve images now so we know the folder before asking further questions
    images, input_folder = resolve_input_images(data_input)
    print(f"Found {len(images)} image(s) in: {input_folder}")

    # --- What to identify ---
    type_of_analysis: str = _get('type_of_analysis') or _get('type-of-analysis') or ''
    if not type_of_analysis and not args.no_interactive:
        type_of_analysis = input(
            '\nWhat should the model identify in each image?\n'
            '  (e.g., "objects", "the architectural style", "the primary land use type")\n'
            '  > '
        ).strip()
    type_of_analysis = type_of_analysis or 'objects'

    prompt_template = (
        "You are a design expert in a design review. You will be shown an image. "
        "Please tell me {what} concisely and only return {what}. "
        "If multiple items are present, separate them with commas. "
        "If you tell me anything other than {what}, you will not be helpful."
    ).format(what=type_of_analysis)

    # --- Number of runs and delay ---
    num_runs_raw = _get('runs')
    if num_runs_raw:
        num_runs: int = int(num_runs_raw)
    elif not args.no_interactive:
        try:
            num_runs = int(input('\nHow many times should each image be analyzed? [1]: ').strip() or '1')
        except Exception:
            num_runs = 1
    else:
        num_runs = 1

    switch_delay: float = _get('delay') or 0.0
    if len(models_to_run) > 1 and switch_delay == 0.0 and not args.no_interactive:
        delay_input = input("\nDelay in seconds between switching models (Enter for 0): ").strip()
        try:
            switch_delay = float(delay_input) if delay_input else 0.0
        except Exception:
            switch_delay = 0.0

    # --- Within-model consensus ---
    within_model = _get('within-model-consensus')
    if within_model is None:
        if args.no_interactive:
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
    between_model = _get('between-model-consensus')
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
        thr_b = input(f"  Fuzzy threshold [{between_model_fuzzy}]: ").strip()
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
    append_metadata = _get('append-metadata')
    if append_metadata is None:
        if args.no_interactive:
            append_metadata = True
        else:
            append_metadata = input(
                "\nAppend run settings and metadata to the Excel output? (y/n) [y]: "
            ).strip().lower() in ('', 'y', 'yes')
    else:
        append_metadata = append_metadata in (True, 'y', 'yes', '1')

    timeout_sec: float = _get('timeout') or 120.0
    max_retries: int = _get('retries') or 2

    # --- Output path ---
    output_file_path = resolve_output_path(input_folder, data_output, len(images), num_runs, len(models_to_run))
    output_dir = os.path.dirname(output_file_path) or '.'
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput will be saved to: {output_file_path}")
    except Exception as e:
        print(f"Warning: could not create output directory {output_dir}: {e}")

    # --- Analysis ---
    master_df = pd.DataFrame({"Image": images})
    metadata_models: list[str] = []
    analysis_start = time.time()

    client = ollama.Client(timeout=timeout_sec)

    for idx, model in enumerate(models_to_run):
        print(f"\nRunning model: {model} ({idx + 1}/{len(models_to_run)})")
        metadata_models.append(model)
        rows = run_model_on_images(client, model, images, input_folder, prompt_template, num_runs, max_retries=max_retries)
        model_df = pd.DataFrame(rows)

        response_cols = [c for c in model_df.columns if c.lower().startswith('response')]
        renamed = {c: f"{c} ({model})" for c in response_cols}
        model_df = model_df.rename(columns=renamed)
        master_df = master_df.merge(model_df, on='Image', how='left')

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

    try:
        master_df = reorder_columns_posthoc(master_df, models_to_run, num_runs)
    except Exception as e:
        print(f"Could not reorder columns: {e}")

    master_df.to_excel(output_file_path, index=False)
    analysis_end = time.time()
    analysis_duration = analysis_end - analysis_start
    print(f"\nAnalysis complete. Results saved to {output_file_path}")

    # --- Post-hoc: aggregate consensus (independent from between-model) ---
    try:
        try:
            df_report = pd.read_excel(output_file_path)
        except Exception as e:
            print(f"Could not read {output_file_path} for post-processing: {e}")
            df_report = None

        agg_summary: dict = {}
        ran_aggregated = False

        if aggregated_choice and df_report is not None:
            resp_cols = [col for col in df_report.columns if col.lower().startswith('response')]
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
                    df_report.to_excel(output_file_path, index=False)
                    print(f"Aggregate consensus written to {output_file_path}")
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
                    df_report.to_excel(output_file_path, index=False)
                    print(f"Between-model consensus saved to {output_file_path}")
                except Exception as e:
                    print(f"Between-model consensus failed: {e}")
            else:
                print("No per-model Consensus columns found for between-model consensus.")

        # --- Metadata ---
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
                ws.append([f"Delay between model runs: {switch_delay}s"])
                ws.append([f"Timeout: {timeout_sec}s, Retries: {max_retries}"])
                ws.append([f"Within-model consensus: {within_model} (mode: {within_model_mode})"])
                if within_model_mode == 'fuzzy':
                    ws.append([f"Within-model fuzzy threshold: {within_model_fuzzy}"])
                ws.append([f"Between-model consensus: {between_model} (mode: {between_model_mode})"])
                if between_model_mode == 'fuzzy':
                    ws.append([f"Between-model fuzzy threshold: {between_model_fuzzy}"])
                ws.append([f"Aggregate consensus: {aggregated_choice} (mode: {agg_mode})"])
                if agg_mode == 'fuzzy':
                    ws.append([f"Aggregate fuzzy threshold: {agg_fuzzy}"])
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
                            id_display = row.get('Image', row_idx + 1)
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

                wb.save(output_file_path)
                print(f"Metadata appended to {output_file_path}")
            except Exception as e:
                print(f"Could not append metadata: {e}")

    except Exception as e:
        print(f"Post-processing error: {e}")


if __name__ == '__main__':
    main()
