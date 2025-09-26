"""Small test harness to exercise within-model (fuzzy) and between-model consensus logic
without invoking Ollama or any long-running tasks.

Run this from the repo root: python run_local_consensus_test.py
"""
import pandas as pd
from image_analysis import compute_consensus_for_block

print('Starting local consensus tests...')

# Create a DataFrame that mimics two models each with 2 replicate responses per image
# Model names used in column labels match the script's expected naming convention.
rows = [
    {
        'Image': 'img1',
        'Response_1 (gemma3:12b)': 'chair, table',
        'Response_2 (gemma3:12b)': 'Chair and table',
        'Response_1 (llava:13b)': 'chair, lamp',
        'Response_2 (llava:13b)': 'chair; lamp'
    },
    {
        'Image': 'img2',
        'Response_1 (gemma3:12b)': 'red car',
        'Response_2 (gemma3:12b)': 'red automobile',
        'Response_1 (llava:13b)': 'car',
        'Response_2 (llava:13b)': 'vehicle'
    },
    {
        'Image': 'img3',
        'Response_1 (gemma3:12b)': 'tree',
        'Response_2 (gemma3:12b)': 'trees',
        'Response_1 (llava:13b)': 'plant',
        'Response_2 (llava:13b)': 'tree'
    },
    {
        'Image': 'img4',
        'Response_1 (gemma3:12b)': '',
        'Response_2 (gemma3:12b)': '',
        'Response_1 (llava:13b)': 'Error: something went wrong',
        'Response_2 (llava:13b)': ''
    }
]

df = pd.DataFrame(rows)
print('\nInput DataFrame:')
print(df)

# Within-model consensus for gemma3
gem_cols = ['Response_1 (gemma3:12b)', 'Response_2 (gemma3:12b)']
consensus_gem, conf_gem = compute_consensus_for_block(df, gem_cols, mode='fuzzy', fuzzy_threshold=80)
print('\nWithin-model (gemma3) fuzzy consensus:')
for img, c, conf in zip(df['Image'], consensus_gem, conf_gem):
    print(f'  {img}: {c} (conf={conf})')

# Within-model consensus for llava
llava_cols = ['Response_1 (llava:13b)', 'Response_2 (llava:13b)']
consensus_llava, conf_llava = compute_consensus_for_block(df, llava_cols, mode='fuzzy', fuzzy_threshold=80)
print('\nWithin-model (llava) fuzzy consensus:')
for img, c, conf in zip(df['Image'], consensus_llava, conf_llava):
    print(f'  {img}: {c} (conf={conf})')

# Now simulate between-model consensus by using the within-model consensus columns
agg_df = df.copy()
agg_df['Consensus (gemma3:12b)'] = consensus_gem
agg_df['Consensus (llava:13b)'] = consensus_llava
between_cols = ['Consensus (gemma3:12b)', 'Consensus (llava:13b)']
between_cons, between_conf = compute_consensus_for_block(agg_df, between_cols, mode='fuzzy', fuzzy_threshold=80)
print('\nBetween-model fuzzy consensus:')
for img, c, conf in zip(agg_df['Image'], between_cons, between_conf):
    print(f'  {img}: {c} (conf={conf})')

print('\nLocal consensus tests complete.')
