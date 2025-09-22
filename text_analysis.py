import pandas as pd
from tqdm import tqdm
from ollama import chat
from ollama import ChatResponse
import os
#!/usr/bin/env python
"""
text_analysis.py - A small wrapper to run text analysis across a CSV/XLSX file using Ollama.

Behavior:
- Prompts for model, prompt, input/output folders, columns and number of runs
- Calls Ollama chat API per row per run, collects responses, and writes an Excel output
"""
import os
import sys
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

    model = input('Model to use (Enter for gemma2): ').strip() or 'gemma2'
    prompt_desc = input('What should the model identify? ').strip() or 'the main topic'
    prompt_template = f'Identify {prompt_desc} in the following text. Return only the identified item(s):'

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

    outname = os.path.splitext(f)[0] + '_text_analysis.xlsx'
    outpath = os.path.join(data_out, outname)
    pd.DataFrame(results).to_excel(outpath, index=False)
    print('Wrote', outpath)


if __name__ == '__main__':
    main()