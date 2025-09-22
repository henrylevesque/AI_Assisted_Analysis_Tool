import os
import sys
import time
import re
import platform
import subprocess
from collections import Counter

import pandas as pd
from tqdm import tqdm
from ollama import chat


def list_input_file(folder):
    return next((f for f in os.listdir(folder) if f.endswith('.csv') or f.endswith('.xlsx')), None)


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
    multi = input("Do you want to compare multiple models? (y/n): ").strip().lower() == 'y'
    models_to_run = []
    if multi:
        print("Enter model names to compare separated by commas, or press Enter to use suggested models:")
        m_input = input().strip()
        if not m_input:
            if models:
                models_to_run = models
            else:
                models_to_run = suggested or ['gemma3:12b', 'gpt-oss:20b']
        else:
            models_to_run = [m.strip() for m in m_input.split(',') if m.strip()]
    else:
        single = input('Enter a single model to use (or press Enter for gemma2): ').strip()
        single_model = single or 'gemma2'
        if models and single_model not in models:
            print(f"Warning: '{single_model}' not found in ollama list; proceeding with provided name.")
        models_to_run = [single_model]

    prompt_desc = input('Enter what you want the model or models to identify within the text').strip() or 'the main topic'
    prompt_template = f'I am going to give you a chunk of text. Please identify {prompt_desc} used in the text.Do not tell me anything besides {prompt_desc} If you tell me anything besides {prompt_desc} you will not be helptful. The text is:'
    
    data_in = input('Data input folder [.] : ').strip() or '.'
    data_out = input('Data output folder [.] : ').strip() or '.'

    f = list_input_file(data_in)
    if not f:
        print('No input CSV/XLSX found in', data_in)
        return
    path = os.path.join(data_in, f)
    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)

    id_col, content_col, num_runs = get_user_columns(df)
    if id_col and id_col in df.columns:
        ids = df[id_col].tolist()
    else:
        ids = list(range(1, len(df) + 1))
    if content_col not in df.columns:
        print('Content column not found. Exiting.')
        return

    contents = df[content_col].tolist()

  
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

    # delay between model runs (useful if switching models manually)
    switch_delay = 0.0
    if len(models_to_run) > 1:
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

    def run_model_on_texts(model_name, texts, prompt_template, num_runs):
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
        rows = run_model_on_texts(model, contents, prompt_template, num_runs)
        model_df = pd.DataFrame(rows)

        # rename response columns to include model name
        response_cols = [c for c in model_df.columns if c.lower().startswith('response')]
        renamed = {c: f"{c} ({model})" for c in response_cols}
        model_df = model_df.rename(columns=renamed)

        # merge responses into master_df by Identifier
        master_df = master_df.merge(model_df, on='Identifier', how='left')

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

    # Append metadata to Excel
    try:
        from openpyxl import load_workbook
        wb = load_workbook(outpath)
        ws = wb.active
        ws.append([])
        ws.append(["Prompt used:", prompt_template])
        ws.append(["Models used:", ', '.join(metadata_models)])
        ws.append([f"Runs per row: {num_runs}"])
        ws.append([f"Delay between model runs: {switch_delay} seconds"])
        ws.append([f"Consensus enabled: {do_consensus}"])
        ws.append([f"Consensus mode: {consensus_mode}"])
        if consensus_mode == 'fuzzy':
            ws.append([f"Fuzzy threshold: {fuzzy_threshold}"])
        hours, rem = divmod(analysis_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        ws.append([f"Duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s"])
        wb.save(outpath)
        print("Reporting metadata appended.")
    except Exception as e:
        print(f"Could not append reporting metadata: {e}")


if __name__ == '__main__':
    main()