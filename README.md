# AI Assisted Analysis Tool

## Introduction

AI Assisted Analysis Tool is an open-source, locally-run toolkit for AI-assisted text and image analysis based on Ollama. It supports three main workflows (text, image, Zotero abstracts) and is designed for reproducible, researcher-friendly analyses. The code allows researchers to run large text based datasets, image datasets, or abstracts exported from Zotero, and run flexible AI enabled analysis on each item. The code logic supports using any LLM from Ollama, and uses a strategy of multiple runs of each item through the LLM that are then consolidated through three main consensus modes to give the modal response across runs to account for LLM errors or hallucinations, with a confidence score based on the percentage of modal responces to total responses. The code optionally supports running through multiple LLM models on the same dataset and allows comparison and consensus calculation within and between models.

See the License and Citation sections for more details: [License](#license) · [Citation](#citation).

Key points:
- Supported inputs: Excel, CSV, image folders, and Zotero exports.
- Configuration: analysis scripts accept YAML or JSON config files (e.g., configs/text_analysis.yaml or configs/image_analysis.json).
- Command-line usage: scripts also accept standard CLI arguments (example flags: --config, --models, --runs, --within-model-consensus, --between-model-consensus, --output). Command-line arguments override config file values.
- Defaults and precedence: built-in defaults → config file → explicit CLI arguments.

Quick examples:
- Use a YAML config:
    ```sh
    python text_analysis.py --config configs/text_analysis.yaml
    ```
- Use a JSON config and override runs on the CLI:
    ```sh
    python image_analysis.py --config configs/image_analysis.json --runs 3
    ```

Why use configs and CLI options:
- Reproducibility: store full run settings in a config file for later reference.
- Automation: enable batch runs or CI by supplying a single config file.
- Flexibility: tweak individual settings on the fly via CLI without editing files.

See the usage sections for each workflow for full lists of accepted config keys and CLI flags (Text Analysis, Image Analysis, Zotero Abstracts). For reporting, outputs include Excel files with optional embedded metadata and a metadata sheet documenting prompt, model, runs, duration, and environment.

## 1. Analysis Workflow - Text

**Purpose:** Analyze any tabular data (Excel or CSV) using AI, not limited to abstracts. This workflow is flexible and user-friendly, allowing you to select which columns to analyze and how the results are reported.

**Key Features:**
- Works with any Excel or CSV file (not just abstracts)
- Lists all columns and lets you select identifier and content columns by name
- Lets you define what you want the AI to identify in your data (custom prompt)
- Allows you to set the number of analysis runs for each row
- Optionally aggregates AI responses for consensus and confidence
- Optionally appends all reporting info (prompt, LLM, runs, hardware, consensus summary, etc.) to the bottom of the Excel output file
- Enhanced reporting: includes prompt, LLM used, number of runs, analysis duration, CPU/GPU info
- Cross-platform: Windows, macOS, Linux

**How to Use:**
1. Prepare your Excel or CSV file and place it in your chosen data input folder.
2. Make sure you also have a data output folder for your file to be saved in after the analysis.
3. Run the script:
    ```powershell
    python text_analysis.py
    ```
4. Follow the prompts:
   - Select the AI model (or press Enter for the recommended model)
   - Enter what you want the program to identify in your data (custom prompt)
   - Choose columns for identifier and content by name
   - Set the number of runs for analysis
   - Optionally aggregate AI responses for consensus and confidence
   - Optionally append all reporting info to the bottom of the output Excel file
5. Review your results in the output Excel file (includes consensus columns and reporting info if selected)

## 2. Analysis Workflow — Image

**Purpose:** Analyze images using local vision-capable models and compare responses across models.

**Key Features:**
- Run one or more vision models sequentially to avoid constant context switching
- Run multiple replicates per image and record within-model Response_1..N
- Compute within-model Consensus and Consensus_Confidence using modes: `exact`, `set`, or `fuzzy`
- Fuzzy consensus uses `rapidfuzz` to cluster similar responses (install `rapidfuzz` to enable)
- Progress bars (tqdm) and an optional inter-model `switch_delay` to allow operator/model switching
- Output is an Excel file with a metadata sheet containing prompt, model(s), runs, duration, and environment info

**How to Use:**
1. Prepare a folder containing your image files and create an output folder for results.
2. Make sure your local Ollama runtime has a vision-capable model available (for example: `gemma23:12b`). Pull it if needed:
    ```powershell
    ollama pull gemma3:12b
    ```
3. Run the script:
    ```powershell
    python image_analysis.py
    ```
4. Follow the prompts:
    - Select the vision model to use (or press Enter for the recommended model)
    - Set the number of runs per image (replications)
    - Choose a consensus mode: `exact`, `set`, or `fuzzy` (set uses normalized sets, fuzzy groups similar responses)
    - If using `fuzzy`, set a fuzzy threshold (default shown in the prompt) and ensure `rapidfuzz` is installed
    - Optionally set `switch_delay` (seconds) to pause between models when running multiple models sequentially
    - Choose whether to append reporting metadata to the output Excel file
5. Inspect the Excel output: each model has Response_1..N columns followed by `Consensus` and `Consensus_Confidence`; a `metadata` sheet contains run details.

**Notes:**
- Install Python dependencies listed in `requirements.txt`. For fuzzy consensus, ensure `rapidfuzz` is present:
  ```powershell
  pip install rapidfuzz
  ```
- PowerShell users: run the commands above in an activated virtual environment to ensure packages are available.


## 3. Zotero Abstracts Workflow - Text

**Purpose:** Analyze bibliographic abstracts exported from Zotero. This workflow is designed for users working specifically with Zotero data and abstracts.

**Key Features:**

**How to Use:**
1. Export your collection from Zotero as a CSV or Excel file
2. Use the provided scripts (e.g., `methods.py`, `results.py`, `location.py`, `theory.py`, `n_themes.py`) in the `python_for_Zotero_abstracts` folder to run targeted analyses on your exported data
3. Follow the prompts in each script for results and consensus aggregation
4. Review your results in the output Excel file

### Prompts

The following scripts in `python_for_Zotero_abstracts` are designed for specific types of analysis:

- `theory.py`: Identify urban planning theory used in abstracts
- `n_themes.py`: Identify three themes from abstracts
- `methods.py`: Identify methods used in abstracts
- `results.py`: Identify results from abstracts
- `location.py`: Identify where the research was conducted


## Requirements
   - `ollama` - Python client for Ollama
      - [Ollama for Windows](https://ollama.com/download/windows)
      - [Ollama for Mac](https://ollama.com/download/mac)
      - [Ollama for Linux](https://ollama.com/download/linux)
  - `pandas` - Data manipulation and analysis
  - `tqdm` - Progress bars
  - `openpyxl` - Excel file handling


### Installation
1. Clone the repository:
    ```powershell
    git clone https://github.com/hleve/AI_Assisted_Analysis_Tool.git
    ```
2. Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```

## Project Logic and Methodology

This tool is designed to leverage AI models for research analysis, with a focus on reliability and reproducibility. The core logic includes:

- **Replication for Error Minimization:**
    - Analyses are run multiple times (replicated) to reduce the impact of random errors or outlier responses from AI models.
    - Aggregation and consensus algorithms are used to combine results, improving reliability and confidence in findings.

- **Prompt Engineering for Quantitative Analysis:**
    - Prompts can be crafted to request numerical values from AI models (e.g., ratings, counts, scores).
    - These outputs can be collected across replications and subjected to standard statistical analyses, such as ANOVA or T-tests, to assess differences, trends, or significance.

This approach allows researchers to harness the flexibility of AI while maintaining scientific rigor and transparency in their workflows.

Each script will prompt you for the required input and provide results in the output Excel file.

## Flow Diagram

```mermaid
graph LR
    A[Start] --> B[Read CSV/Excel File/Image folder]
    B --> C[Initialize Responses List]
    C --> D{Loop Through Runs}
    D --> E[Run 1]
    D --> F[Run 2]
    D --> G[Run N]
    E --> H{Loop Through Rows}
    F --> H
    G --> H
    H --> I[Append Title and Abstract]
    I --> J[Parse prompt]
    J --> K[Send to AI Model(s)]
    K --> L[Receive Response]
    L --> M[Append Response to List]
    M --> H
    H --> N[Ensure Correct Number of Elements]
    N --> O[Create DataFrame]
    O --> P[Write to Excel File]
    P --> Q[End]
```

## Directions

1. **Navigate to the parent directory**:
    ```sh
    cd /path/to/your/directory
    ```

2. **Create the virtual environment**:
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install Requirements**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Install Chosen LLM (gemma2 is recommended for text, and gemma3:12b is recomended for images)**:
    ```powershell
    ollama pull gemma2
    ```

## Troubleshooting

### Import Errors
If you encounter import errors for `ollama`, `pandas`, or `tqdm`, ensure that:

1. **Virtual environment is activated**: Make sure you've activated your virtual environment before installing packages or running scripts.
2. **Correct package installation**: Run `pip install -r requirements.txt` in your activated virtual environment.
3. **VS Code Python interpreter**: If using VS Code, ensure it's using the Python interpreter from your virtual environment:
   - Open Command Palette (Ctrl+Shift+P)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from your `venv` folder

### Common Package Issues
- **ollama package**: The correct package name is `ollama`, not `ollama_python`
- **Excel support**: `openpyxl` is required for writing Excel files with pandas
- **Progress bars**: `tqdm` provides the progress bars shown during analysis

### Virtual Environment Issues on Windows
If you encounter PowerShell execution policy errors when activating the virtual environment:
```powershell
# Use the batch file instead
.\venv\Scripts\activate.bat

# Or bypass execution policy temporarily
powershell -ExecutionPolicy Bypass -File .\venv\Scripts\Activate.ps1
```

## Cloning the Repository

1. **Open a terminal or command prompt**.
2. **Navigate to the directory where you want to clone the repository**:
    ```sh
    cd /path/to/your/directory
    ```
3. **Clone the repository**:
    ```sh
    git clone https://github.com/hleve/AI_Analysis_Tool.git
    ```
4. **Navigate to the cloned repository directory**:
    ```sh
    cd AI_Analysis_Tool
    ```

## Running a Python File

1. **Ensure the virtual environment is activated**:
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

2. **Run the desired Python file**:
        ```sh
        python <filename>.py
        ```
        Replace `<filename>` with the name of the Python file you want to run. For example, to run `text_analysis.py`, use:
        ```sh
        python text_analysis.py
        ```

        You can run the scripts three ways:

        - Interactive CLI prompts (bypass config):

            Run the script without `--no-interactive` or `--config`. The script will ask you for any missing settings (model, input, output, columns, runs, etc.). Example:

            ```powershell
            python text_analysis.py
            # follow interactive prompts to select model, input file, columns, and runs
            ```

        - Fully non-interactive via CLI arguments (no config file):

            Provide all required settings on the command line and include `--no-interactive` to prevent prompts. CLI arguments override values in any config file. Example:

            ```powershell
            python image_analysis.py --models "gemma3:12b" --input "./images" --output "results.xlsx" --runs 2 --within-model-consensus --within-model-consensus-mode fuzzy --within-model-fuzzy-threshold 85 --no-interactive
            ```

        - Config-file driven (JSON or YAML) with optional CLI overrides:

            Supply a `--config` file in JSON or YAML format. The script will read settings from that file. Any CLI arguments you pass will override corresponding config values. Use `--no-interactive` for fully deterministic runs.

            ```powershell
            python text_analysis.py --config configs/text_config_example.yaml --no-interactive
            # or override a setting from the config:
            python text_analysis.py --config configs/text_config_example.yaml --runs 1 --no-interactive
            ```
            
            ### Boolean flags: within-model / between-model consensus and append-metadata

            The scripts use mutually-exclusive on/off flags for important boolean options so that the absence of a flag does not accidentally override a value in your config file. The relevant tri-state flags are:

            - Within-model consensus: `--within-model-consensus` / `--no-within-model-consensus` (defaults to ON when not specified)
+            - Between-model consensus: `--between-model-consensus` / `--no-between-model-consensus` (defaults to ON when not specified)
+            - Append metadata: `--append-metadata` / `--no-append-metadata` (defaults to ON when not specified)
+
            Specifying the `--within-model-consensus` flag forces within-model consensus on for the run; `--no-within-model-consensus` forces it off. Omitting both will use the config file value or the script default.

3. **Follow any additional prompts or instructions** provided by the script to complete the analysis.

### Getting Started
See `documentation.md` for a step-by-step guide.

### Contributing
Contributions are welcome! Please see `CONTRIBUTING.md` for contribution guidelines, issue templates, and the code of conduct.


## License
See `LICENSE` for details.

## Citation

If you use this software (or parts of it) in a publication, please cite this project. The canonical citation information is included in the repository's `CITATION.cff` file — please use that metadata (authors, title, version, DOI) when referencing this work.

### Example citations

Using the metadata in `CITATION.cff`, here are two example citation formats you can copy:

- APA:

    Levesque, H. (2025). AI_Assisted_Analysis_Tool (version 1.2-beta) [Software]. Zenodo. https://doi.org/10.5281/zenodo.14932653

- BibTeX:

    ```bibtex
    @software{levesque_ai_2025,
        author = {Levesque, Henry},
        title = {AI_Assisted_Analysis_Tool},
        year = {2025},
        version = {1.2-beta},
        doi = {10.5281/zenodo.14932653},
        url = {https://github.com/henrylevesque/AI_Analysis_Tool}
    }
    ```