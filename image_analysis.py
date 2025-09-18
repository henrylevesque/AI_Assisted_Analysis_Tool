import os
import time
import platform
import pandas as pd
from tqdm import tqdm
from ollama import chat
import sys


def list_image_files(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]


def main():
    print("Python executable:", sys.executable)

    # Check models (best-effort)
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        models = []
        for line in result.stdout.splitlines():
            if line.strip() and not line.lower().startswith("name"):
                models.append(line.split()[0])
    except Exception:
        models = ["gemma3"]

    selected_model = input("Enter the name of the model to use (or press Enter for the recommended vision model gemma3): ").strip()
    if not selected_model or selected_model not in models:
        selected_model = "gemma3"

    print(f"Using model: {selected_model}\n")

    # Prompt / analysis instruction
    type_of_analysis = input("Enter what you want the program to identify within the image(s) (e.g., objects, scene, text): ").strip()
    prompt_template = (
        "You will be shown an image. Please describe the {what} present in the image concisely and only return the {what}. "
        "If multiple items are present, separate them with commas."
    )

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
    output_file_name = f"image_analysis_{len(images)}images_{num_runs}runs.xlsx"
    output_file_path = os.path.join(data_output_folder, output_file_name)

    analysis_start = time.time()

    all_results = []
    for idx, img in enumerate(images, 1):
        img_path = os.path.join(data_input_folder, img)
        print(f"\nProcessing image {idx}/{len(images)}: {img}")
        row_responses = []

        for run in tqdm(range(num_runs), desc="Running analysis", leave=False):
            try:
                # Call the multimodal model with the image
                msg = {
                    "role": "user",
                    "content": prompt_template.format(what=type_of_analysis),
                    "images": [img_path],
                }
                response = chat(model=selected_model, messages=[msg])
                cleaned = response['message']['content'].strip().replace('\r', ' ').replace('\n', ' ')
                row_responses.append(cleaned)
            except Exception as e:
                print(f"Error on image {img}, run {run + 1}: {e}")
                row_responses.append("Error occurred")

        result = {"Image": img}
        for i, r in enumerate(row_responses, 1):
            result[f"Response_{i}"] = r
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_excel(output_file_path, index=False)

    analysis_end = time.time()
    analysis_duration = analysis_end - analysis_start
    print(f"\nAnalysis complete. Results saved to {output_file_path}")

    # Aggregation / consensus
    run_aggregation = input("Do you want to aggregate AI responses for consensus? (y/n): ").strip().lower()
    if run_aggregation == 'y':
        df = pd.read_excel(output_file_path)
        import re
        from collections import Counter

        response_cols = [c for c in df.columns if c.lower().startswith('response')]
        df['Consensus_Result'] = ''
        df['Consensus_Confidence'] = 0.0

        for i, row in df.iterrows():
            responses = [str(row[c]) for c in response_cols if pd.notna(row[c])]
            if not responses:
                df.at[i, 'Consensus_Result'] = 'no responses'
                df.at[i, 'Consensus_Confidence'] = 0.0
                continue
            normalized = [re.sub(r"\s+", ' ', r.lower().strip()) for r in responses]
            counts = Counter(normalized)
            most_common, count = counts.most_common(1)[0]
            confidence = count / len(responses)
            df.at[i, 'Consensus_Result'] = most_common
            df.at[i, 'Consensus_Confidence'] = round(confidence, 3)

        # Save updated file with consensus
        df.to_excel(output_file_path, index=False)
        print(f"Consensus results appended to {output_file_path}")

        # Optional: append reporting metadata to bottom of Excel
        add_reporting = input("Add reporting metadata to bottom of the Excel file? (y/n): ").strip().lower()
        if add_reporting == 'y':
            try:
                from openpyxl import load_workbook
                wb = load_workbook(output_file_path)
                ws = wb.active
                ws.append([])
                ws.append(["Prompt used:", prompt_template.format(what=type_of_analysis)])
                ws.append([f"Model: {selected_model}"])
                ws.append([f"Runs per image: {num_runs}"])
                hours, rem = divmod(analysis_duration, 3600)
                minutes, seconds = divmod(rem, 60)
                ws.append([f"Duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s"])
                wb.save(output_file_path)
                print("Reporting metadata appended.")
            except Exception as e:
                print(f"Could not append reporting metadata: {e}")


if __name__ == '__main__':
    main()
