import os
import time
import pandas as pd
from tqdm import tqdm
from ollama import chat
import sys
import re
from collections import Counter


def list_image_files(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]


def run_model_on_images(model_name, images, data_input_folder, prompt_template, num_runs):
    """Run a single model across images and return a list of dict rows with Response_{i} keys labeled by run index."""
    rows = []
    for idx, img in enumerate(images, 1):
        img_path = os.path.join(data_input_folder, img)
        row_responses = []
        for run in range(num_runs):
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
            # Label responses tentatively; higher-level code will rename to include model
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
    # split on commas and semicolons, strip, normalize tokens, return sorted unique list
    parts = re.split(r"[,;]+", s) if s else []
    normalized = [_normalize_text(p) for p in parts if _normalize_text(p)]
    # remove duplicates while preserving order
    seen = set()
    out = []
    for x in normalized:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _fuzzy_group_responses(responses, threshold=85):
    # lazy import to avoid hard dependency unless used
    try:
        from rapidfuzz import fuzz
    except Exception as e:
        raise RuntimeError("rapidfuzz is required for fuzzy grouping; please install it (pip install rapidfuzz)") from e

    groups = []  # list of lists
    for r in responses:
        placed = False
        for g in groups:
            # compare to representative of group (first element)
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
    """
    Compute consensus and confidence for a block of response columns.

    mode: 'exact' | 'set' | 'fuzzy'
      - exact: majority vote on normalized full-string responses
      - set: treat each response as a comma-separated set; consensus is the most common item(s) across runs (returns comma-joined consensus)
      - fuzzy: group similar strings using fuzzy matching (rapidfuzz) then majority vote within groups

    Returns (consensus_list, confidences_list)
    """
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
            # aggregate items across responses
            all_items = []
            for r in responses:
                items = _split_and_normalize_set(r)
                all_items.extend(items)
            if not all_items:
                consensus.append('')
                confidences.append(0.0)
                continue
            counts = Counter(all_items)
            # choose items that appear in >50% of responses by default
            chosen = [it for it, ct in counts.items() if ct / len(responses) >= 0.5]
            if not chosen:
                # fallback: pick most common
                chosen = [counts.most_common(1)[0][0]]
            consensus.append(', '.join(chosen))
            # compute a simple confidence: average frequency of chosen items / num responses (capped at 1)
            avg_freq = sum(counts[it] for it in chosen) / (len(chosen) * len(responses))
            confidences.append(round(avg_freq, 3))

        elif mode == 'fuzzy':
            # group responses by fuzzy similarity
            normalized = [_normalize_text(r) for r in responses]
            groups = _fuzzy_group_responses(normalized, threshold=fuzzy_threshold)
            # pick largest group
            groups_sorted = sorted(groups, key=lambda g: len(g), reverse=True)
            top_group = groups_sorted[0]
            # representative consensus: most common string within the top group
            counts = Counter(top_group)
            rep, count = counts.most_common(1)[0]
            confidence = len(top_group) / len(responses)
            consensus.append(rep)
            confidences.append(round(confidence, 3))

        else:
            raise ValueError(f"Unknown consensus mode: {mode}")

    return consensus, confidences


def reorder_columns_posthoc(df, models, num_runs):
    """Return a dataframe with columns reordered so that all Response_* columns come first (grouped by run index across models),
    followed by Consensus and Consensus_Confidence columns grouped per model at the end.

    Order: Image, Response_1 (model1), Response_1 (model2), ..., Response_2 (model1), Response_2 (model2), ...,
    then Consensus (model1), Consensus_Confidence (model1), Consensus (model2), Consensus_Confidence (model2), ...
    """
    # Base column
    cols = ['Image']

    # Add response columns grouped by run index across models
    for run_idx in range(1, num_runs + 1):
        for m in models:
            cols.append(f"Response_{run_idx} ({m})")

    # Then add consensus and confidence grouped per model
    for m in models:
        cols.append(f"Consensus ({m})")
        cols.append(f"Consensus_Confidence ({m})")

    # Filter to existing columns only (in case some models had missing runs)
    existing = [c for c in cols if c in df.columns]
    # Add any other columns (unexpected) at the end to avoid data loss
    remaining = [c for c in df.columns if c not in existing]
    final_order = existing + remaining
    return df[final_order]


def main():
    print("Python executable:", sys.executable)

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

    # Suggested vision models (user-provided list could be inserted here)
    suggested = ["gemma3:12b", "llava:13b", "llama3.2-vision:11b", "qwen2.5vl:7b", "bakllava:7b"]

    print("Discovered models:", discovered or '<none>')
    print("Suggested vision models:", suggested)
import os
import time
import pandas as pd
from tqdm import tqdm
from ollama import chat
import sys
import re
from collections import Counter


def list_image_files(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]


def run_model_on_images(model_name, images, data_input_folder, prompt_template, num_runs):
    """Run a single model across images and return a list of dict rows with Response_{i} keys labeled by run index."""
    rows = []
    for idx, img in enumerate(images, 1):
        img_path = os.path.join(data_input_folder, img)
        row_responses = []
        for run in range(num_runs):
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
            # Label responses tentatively; higher-level code will rename to include model
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

    multi = input("Compare multiple models? (y/n): ").strip().lower() == 'y'
    models_to_run = []
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

    # Consensus selection
    do_consensus = input("Compute consensus after runs? (y/n) [y]: ").strip().lower()
    do_consensus = True if do_consensus in ('', 'y', 'yes') else False
    consensus_mode = 'exact'
    fuzzy_threshold = 85
    if do_consensus:
        print("Consensus modes:\n  exact - normalized exact majority (default)\n  set - normalized set voting for comma-separated lists\n  fuzzy - fuzzy-string grouping (requires rapidfuzz)")
        mode_in = input("Choose consensus mode (exact/set/fuzzy) [exact]: ").strip().lower() or 'exact'
        if mode_in in ('exact', 'set', 'fuzzy'):
            consensus_mode = mode_in
        else:
            print("Unknown mode, using 'exact'.")
        if consensus_mode == 'fuzzy':
            thr = input("Fuzzy threshold (0-100) [85]: ").strip()
            try:
                fuzzy_threshold = int(thr) if thr else 85
            except Exception:
                fuzzy_threshold = 85

    # Prompt / analysis instruction
    type_of_analysis = input("Enter what you want the program to identify within the image(s) (e.g., objects, scene, text): ").strip()
    prompt_template = (
        "You will be shown an image. You are a design expert in a design review. Please describe {what} concisely and only return {what}. "
        "If multiple items are present, separate them with commas."
    ).format(what=type_of_analysis)

    # Input/output folders
    data_input_folder = input("Enter the path to the image input folder: ").strip()
    data_output_folder = input("Enter the path to the data output folder: ").strip()
    if not os.path.isdir(data_input_folder):
        raise FileNotFoundError("Input folder not found.")
    if not os.path.isdir(data_output_folder):
        os.makedirs(data_output_folder, exist_ok=True)

    images = list_image_files(data_input_folder)
    if not images:
        raise FileNotFoundError("No image files found in the specified input folder.")

    num_runs = int(input("Enter number of times to run analysis per image: ").strip())
    delay_input = input("Enter delay between model runs in seconds (e.g., 1.0) or press Enter for 0: ").strip()
    switch_delay = float(delay_input) if delay_input else 0.0
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



if __name__ == '__main__':
    main()
