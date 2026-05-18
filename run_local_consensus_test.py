"""Local consensus test harness — no Ollama or network access required.

Tests the consensus logic in both image_analysis.py and text_analysis.py.

Run from the repo root:
    python run_local_consensus_test.py
"""
import sys
import pandas as pd
from image_analysis import compute_consensus_for_block as img_consensus
from text_analysis import compute_consensus_for_block as txt_consensus

PASS = "PASS"
FAIL = "FAIL"
errors: list[str] = []

def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS} {label}")
    else:
        msg = f"  {FAIL} {label}" + (f": {detail}" if detail else "")
        print(msg)
        errors.append(msg)


# ---------------------------------------------------------------------------
# Shared test data — mimics two models with 2 runs each per image/row
# ---------------------------------------------------------------------------
rows = [
    {
        'Image': 'img1', 'Identifier': 'r1',
        'Response_1 (gemma3:12b)': 'chair, table',
        'Response_2 (gemma3:12b)': 'Chair and table',
        'Response_1 (llava:13b)': 'chair, lamp',
        'Response_2 (llava:13b)': 'chair; lamp',
    },
    {
        'Image': 'img2', 'Identifier': 'r2',
        'Response_1 (gemma3:12b)': 'red car',
        'Response_2 (gemma3:12b)': 'red automobile',
        'Response_1 (llava:13b)': 'car',
        'Response_2 (llava:13b)': 'vehicle',
    },
    {
        'Image': 'img3', 'Identifier': 'r3',
        'Response_1 (gemma3:12b)': 'tree',
        'Response_2 (gemma3:12b)': 'trees',
        'Response_1 (llava:13b)': 'plant',
        'Response_2 (llava:13b)': 'tree',
    },
    {
        'Image': 'img4', 'Identifier': 'r4',
        'Response_1 (gemma3:12b)': '',
        'Response_2 (gemma3:12b)': '',
        'Response_1 (llava:13b)': 'Error: something went wrong',
        'Response_2 (llava:13b)': '',
    },
]
df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# IMAGE ANALYSIS — within-model fuzzy consensus
# ---------------------------------------------------------------------------
print("\n--- image_analysis: within-model fuzzy consensus ---")
gem_cols = ['Response_1 (gemma3:12b)', 'Response_2 (gemma3:12b)']
cons_gem, conf_gem = img_consensus(df, gem_cols, mode='fuzzy', fuzzy_threshold=80)

check("img1 gemma3 consensus is non-empty", bool(cons_gem[0]), cons_gem[0])
check("img1 gemma3 confidence > 0", conf_gem[0] > 0, str(conf_gem[0]))
check("img4 empty responses yield empty consensus", cons_gem[3] == '', repr(cons_gem[3]))
check("img4 empty responses yield 0.0 confidence", conf_gem[3] == 0.0, str(conf_gem[3]))

llava_cols = ['Response_1 (llava:13b)', 'Response_2 (llava:13b)']
cons_llava, conf_llava = img_consensus(df, llava_cols, mode='fuzzy', fuzzy_threshold=80)
check("img1 llava consensus is non-empty", bool(cons_llava[0]), cons_llava[0])

# ---------------------------------------------------------------------------
# IMAGE ANALYSIS — between-model consensus using new column names
# ---------------------------------------------------------------------------
print("\n--- image_analysis: between-model consensus (Consensus / Confidence columns) ---")
agg_df = df.copy()
agg_df['Consensus (gemma3:12b)'] = cons_gem
agg_df['Confidence (gemma3:12b)'] = conf_gem
agg_df['Consensus (llava:13b)'] = cons_llava
agg_df['Confidence (llava:13b)'] = conf_llava

bm_cons, bm_conf = img_consensus(agg_df, ['Consensus (gemma3:12b)', 'Consensus (llava:13b)'], mode='fuzzy', fuzzy_threshold=80)
check("between-model returns one result per row", len(bm_cons) == len(rows), str(len(bm_cons)))
# img4: gemma3 consensus is '' (both runs empty), llava consensus is non-empty (error string).
# After filtering empty strings, one valid response remains -> confidence 1.0 is correct.
check("between-model img4 has one valid source -> confidence 1.0", bm_conf[3] == 1.0, str(bm_conf[3]))

# ---------------------------------------------------------------------------
# TEXT ANALYSIS — exact mode
# ---------------------------------------------------------------------------
print("\n--- text_analysis: within-model exact consensus ---")
exact_rows = [
    {'Identifier': 'a', 'Response_1 (m1)': 'urban planning', 'Response_2 (m1)': 'Urban Planning', 'Response_3 (m1)': 'urban planning'},
    {'Identifier': 'b', 'Response_1 (m1)': 'housing', 'Response_2 (m1)': 'transport', 'Response_3 (m1)': 'housing'},
    {'Identifier': 'c', 'Response_1 (m1)': '', 'Response_2 (m1)': '', 'Response_3 (m1)': ''},
]
edf = pd.DataFrame(exact_rows)
e_cols = ['Response_1 (m1)', 'Response_2 (m1)', 'Response_3 (m1)']
e_cons, e_conf = txt_consensus(edf, e_cols, mode='exact')

check("row a: majority consensus is 'urban planning'", e_cons[0] == 'urban planning', repr(e_cons[0]))
check("row a: confidence is 1.0 (3/3 after normalisation)", e_conf[0] == 1.0, str(e_conf[0]))
check("row b: 'housing' wins 2/3", e_cons[1] == 'housing', repr(e_cons[1]))
check("row b: confidence is ~0.667", round(e_conf[1], 2) == 0.67, str(e_conf[1]))
check("row c: empty input yields empty consensus", e_cons[2] == '', repr(e_cons[2]))
check("row c: empty input yields 0.0 confidence", e_conf[2] == 0.0, str(e_conf[2]))

# ---------------------------------------------------------------------------
# TEXT ANALYSIS — set mode
# ---------------------------------------------------------------------------
print("\n--- text_analysis: within-model set consensus ---")
set_rows = [
    {'Identifier': 'x', 'Response_1 (m1)': 'housing, transport', 'Response_2 (m1)': 'housing; zoning', 'Response_3 (m1)': 'housing, transport'},
]
sdf = pd.DataFrame(set_rows)
s_cons, s_conf = txt_consensus(sdf, ['Response_1 (m1)', 'Response_2 (m1)', 'Response_3 (m1)'], mode='set')
check("set mode: 'housing' appears in all 3 runs - in consensus", 'housing' in s_cons[0], repr(s_cons[0]))
check("set mode: confidence > 0", s_conf[0] > 0, str(s_conf[0]))

# ---------------------------------------------------------------------------
# TEXT ANALYSIS — fuzzy mode
# ---------------------------------------------------------------------------
print("\n--- text_analysis: within-model fuzzy consensus ---")
fuz_rows = [
    {'Identifier': 'y', 'Response_1 (m1)': 'new urbanism', 'Response_2 (m1)': 'New Urbanism theory', 'Response_3 (m1)': 'new urban theory'},
]
fdf = pd.DataFrame(fuz_rows)
f_cons, f_conf = txt_consensus(fdf, ['Response_1 (m1)', 'Response_2 (m1)', 'Response_3 (m1)'], mode='fuzzy', fuzzy_threshold=70)
check("fuzzy mode: all three cluster together", f_conf[0] > 0.5, str(f_conf[0]))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if errors:
    print(f"FAILED — {len(errors)} test(s) failed:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("All tests passed.")
