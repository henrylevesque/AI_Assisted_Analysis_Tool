# Prompt Library

Ready-to-use `prompt_desc` values for `text_analysis.py` and `type_of_analysis` values for `image_analysis.py`.

Pass any of these directly on the command line or in a config file.

---

## Text Analysis Prompts (`prompt_desc`)

These are designed for use with `text_analysis.py`. The value you supply becomes the fill-in for the prompt template:

> *"Please identify **[prompt_desc]** used in the text. Do not tell me anything besides **[prompt_desc]**."*

### Literature Review / Bibliographic Analysis

| Goal | `prompt_desc` value |
|------|---------------------|
| Identify theoretical framework | `"the planning theory or theoretical framework"` |
| Extract research methodology | `"the research methodology or methods used"` |
| Summarize findings | `"the main findings or results reported"` |
| Identify study location | `"the geographic location or study area"` |
| Extract main themes (single) | `"the primary theme or topic"` |
| Extract main themes (list) | `"the three main themes, separated by commas"` |
| Identify research gap | `"the research gap or problem identified"` |
| Extract keywords | `"five keywords that best describe this text, separated by commas"` |
| Classify study type | `"the type of study (e.g., qualitative, quantitative, mixed methods, review)"` |
| Identify data sources | `"the data sources or datasets used"` |

### Policy and Document Analysis

| Goal | `prompt_desc` value |
|------|---------------------|
| Extract policy goals | `"the main policy goals or objectives"` |
| Identify stakeholders | `"the stakeholders or actors mentioned"` |
| Identify policy instruments | `"the policy instruments or tools proposed"` |
| Extract outcomes | `"the outcomes or impacts described"` |
| Identify barriers | `"the barriers or challenges identified"` |

### General Purpose

| Goal | `prompt_desc` value |
|------|---------------------|
| Topic classification | `"the main topic or subject"` |
| Sentiment | `"the overall sentiment (positive, negative, or neutral)"` |
| Extract named entities | `"the organizations, places, or people mentioned, separated by commas"` |
| Summarize argument | `"the central argument or claim"` |

---

## Image Analysis Prompts (`type_of_analysis`)

These are designed for use with `image_analysis.py`. The value fills into:

> *"Please tell me **[type_of_analysis]** concisely and only return **[type_of_analysis]**."*

### Urban and Built Environment

| Goal | `type_of_analysis` value |
|------|--------------------------|
| Identify land use | `"the primary land use type (e.g., residential, commercial, industrial, park)"` |
| Describe street character | `"the street character and dominant features"` |
| Identify building materials | `"the primary building materials visible"` |
| Count floors | `"the number of floors or storeys of the main building"` |
| Assess greenery | `"the level of greenery or vegetation presence (low, medium, high)"` |
| Identify transport infrastructure | `"the transportation infrastructure visible (e.g., roads, bike lanes, transit stops)"` |

### Design Review

| Goal | `type_of_analysis` value |
|------|--------------------------|
| Overall design style | `"the architectural or design style"` |
| Identify dominant colors | `"the three dominant colors, separated by commas"` |
| Assess visual complexity | `"the visual complexity of the scene (low, medium, high)"` |
| Describe spatial layout | `"the spatial layout and organization"` |

### General Purpose

| Goal | `type_of_analysis` value |
|------|--------------------------|
| Object identification | `"the main objects present, separated by commas"` |
| Scene classification | `"the type of scene or setting"` |
| Activity detection | `"the activities or uses occurring in the image"` |
| Safety assessment | `"potential safety concerns visible in the image"` |

---

## Example Workflows

### Single-model text analysis (terminal)

```bash
python text_analysis.py \
  --input ./data/abstracts.xlsx \
  --id-col id \
  --content-col abstract \
  --runs 5 \
  --models gemma3:12b \
  --no-interactive
```

When prompted (or via `--prompt-desc`), use a value from the table above, e.g.:
`"the planning theory or theoretical framework"`

### Multi-model comparison with config file

```bash
python text_analysis.py --config text_config.yaml --no-interactive
```

In `text_config.yaml`:
```yaml
models:
  - gemma3:12b
  - deepseek-r1:14b
prompt_desc: "the three main themes, separated by commas"
within_model_consensus: true
within_model_consensus_mode: fuzzy
between_model_consensus: true
runs: 5
```

### Image analysis (terminal)

```bash
python image_analysis.py \
  --input ./data/images/ \
  --runs 3 \
  --models llava:13b \
  --type-of-analysis "the primary land use type" \
  --no-interactive
```

---

## Choosing a Consensus Mode

| Mode | Best for | Notes |
|------|----------|-------|
| `exact` | Short, discrete labels (sentiment, land use type) | Normalizes case and punctuation before comparing |
| `set` | Comma-separated lists (themes, keywords) | Items appearing in >50% of runs are kept |
| `fuzzy` | Free-text descriptions that may be paraphrased | Requires `rapidfuzz`; threshold 80–90 recommended |
