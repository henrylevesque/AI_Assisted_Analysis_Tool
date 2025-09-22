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

    model = input('Model to use (Enter for gemma2): ').strip() or 'gemma2'
    if models and model not in models:
        print(f"Warning: '{model}' not found in ollama list; proceeding with provided name.")

    prompt_desc = input('What should the model identify? ').strip() or 'the main topic'
    prompt_template = f'I am going to give you a chunk of text. Please identify {prompt_desc} used in the text. Return only the identified item(s).'

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
    results = []
    analysis_start = time.time()
    for i, (ident, text) in enumerate(zip(ids, contents)):
        print(f'\nRow {i+1}/{len(contents)}')
        rowr = {'Identifier': ident, 'Content': text}
        for r in range(1, num_runs+1):
            try:
                resp = chat(model=model, messages=[{'role': 'user', 'content': f'{prompt_template} {text}'}])
                out = resp['message']['content'].strip().replace('\n', ' ').replace('\r', ' ')
            except Exception as e:
                out = f'Error: {e}'
            rowr[f'Response_{r}'] = out
        results.append(rowr)

    base = os.path.splitext(f)[0]
    outname = f"{base}_text_analysis_{num_runs}runs.xlsx"
    outpath = os.path.join(data_out, outname)
    pd.DataFrame(results).to_excel(outpath, index=False)
    analysis_end = time.time()
    analysis_duration = analysis_end - analysis_start
    print('Wrote', outpath)

    # --- Optional aggregation / consensus ---
    run_aggregation = input("Do you want to aggregate AI responses for consensus? (y/n): ").strip().lower()
    if run_aggregation == 'y':
        rdf = pd.read_excel(outpath)
        response_cols = [c for c in rdf.columns if c.lower().startswith('response')]
        rdf['Consensus_Result'] = ''
        rdf['Consensus_Confidence'] = 0.0
        for idx, row in rdf.iterrows():
            responses = [str(r) for r in row[response_cols].tolist() if pd.notna(r)]
            if not responses:
                rdf.at[idx, 'Consensus_Result'] = 'no responses'
                rdf.at[idx, 'Consensus_Confidence'] = 0.0
                continue
            normalized = [re.sub(r'\s+', ' ', r.lower().strip()) for r in responses]
            counts = Counter(normalized)
            most_common, most_count = counts.most_common(1)[0]
            conf = most_count / len(responses)
            rdf.at[idx, 'Consensus_Result'] = most_common
            rdf.at[idx, 'Consensus_Confidence'] = round(conf, 3)

        # Print summary
        high = rdf[rdf['Consensus_Confidence'] >= 0.7]
        mid = rdf[(rdf['Consensus_Confidence'] >= 0.4) & (rdf['Consensus_Confidence'] < 0.7)]
        low = rdf[rdf['Consensus_Confidence'] < 0.4]
        print('\nConsensus Summary:')
        print(f'High confidence (≥70%): {len(high)} rows')
        print(f'Medium confidence (40-69%): {len(mid)} rows')
        print(f'Low confidence (<40%): {len(low)} rows')
        if len(low) > 0:
            print('\nRows with low confidence may require manual review:')
            for ridx, rrow in low.iterrows():
                print(f"  Row {ridx + 1}: {rrow.get('Identifier', '')} (confidence: {rrow['Consensus_Confidence']:.1%})")

        # System reporting
        print('\nPrompt used for analysis:')
        print(prompt_template)
        print(f'Ollama model used: {model}')
        print(f'Number of runs per row: {num_runs}')
        hours, rem = divmod(analysis_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Analysis duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s')
        try:
            import cpuinfo
            cpu = cpuinfo.get_cpu_info()
            cpu_brand = cpu.get('brand_raw', 'Unknown CPU')
        except Exception:
            cpu_brand = platform.processor() or platform.machine() or 'Unknown CPU'
        print(f'CPU: {cpu_brand}')

        # GPU detection (best-effort)
        gpu_info = None
        integrated_gpu = None
        for brand in ["Radeon", "NVIDIA", "Intel Graphics", "Iris", "GeForce", "RTX", "GTX"]:
            if brand.lower() in cpu_brand.lower():
                integrated_gpu = cpu_brand
                break
        try:
            if sys.platform.startswith('win'):
                gres = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True)
                glines = [l.strip() for l in gres.stdout.splitlines() if l.strip() and 'Name' not in l]
                gpu_info = ', '.join(glines) if glines else None
            elif sys.platform.startswith('linux'):
                gres = subprocess.run('lspci | grep -i vga', shell=True, capture_output=True, text=True)
                glines = [l for l in gres.stdout.splitlines() if l]
                gpu_info = ', '.join(glines) if glines else None
            elif sys.platform == 'darwin':
                gres = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True)
                g = []
                for line in gres.stdout.splitlines():
                    if 'Chipset Model:' in line or 'Vendor:' in line:
                        g.append(line.strip())
                gpu_info = ', '.join(g) if g else None
        except Exception:
            gpu_info = None
        all_gpus = []
        if integrated_gpu:
            all_gpus.append(f"Integrated GPU: {integrated_gpu}")
        if gpu_info:
            all_gpus.append(f"Detected GPU(s): {gpu_info}")
        gpu_report = ', '.join(all_gpus) if all_gpus else 'Not detected'
        print(f'GPU: {gpu_report}')

        # Save consensus columns back to the same output file
        rdf.to_excel(outpath, index=False)
        print(f'\nAnalysis complete with consensus calculation! Results saved to {outpath}')

        add_reporting = input('Do you want to add all reporting information to the bottom of the output file? (y/n): ').strip().lower()
        if add_reporting == 'y':
            try:
                from openpyxl import load_workbook
                wb = load_workbook(outpath)
                ws = wb.active
                ws.append([])
                ws.append(["Consensus Summary:"])
                ws.append([f"High confidence (≥70%): {len(high)} rows"]) 
                ws.append([f"Medium confidence (40-69%): {len(mid)} rows"]) 
                ws.append([f"Low confidence (<40%): {len(low)} rows"]) 
                if len(low) > 0:
                    ws.append(["Rows with low confidence may require manual review:"])
                    for ridx, rrow in low.iterrows():
                        ws.append([f"Row {ridx + 1}: {rrow.get('Identifier', '')} (confidence: {rrow['Consensus_Confidence']:.1%})"])
                ws.append([])
                ws.append(["Prompt used for analysis:"])
                ws.append([prompt_template])
                ws.append([f"Ollama model used: {model}"])
                ws.append([f"Number of runs per row: {num_runs}"])
                ws.append([f"Analysis duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s"])
                ws.append([f"CPU: {cpu_brand}"])
                ws.append([f"GPU: {gpu_report}"])
                wb.save(outpath)
                print(f"\nReporting information added to the bottom of {outpath}")
            except Exception as e:
                print(f"Could not append reporting info: {e}")


if __name__ == '__main__':
    main()