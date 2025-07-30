import pandas as pd
from tqdm import tqdm
from ollama import chat
from ollama import ChatResponse
import os
import argparse

def get_user_columns():
    """Get column selections from user input"""
    id_col = input("Enter the column number to use as identifier (reminder: subtract 1 from the column number) (or press Enter to use auto-numbering): ").strip()
    content_col = input("Enter the column number for content to analyze (reminder: subtract 1 from the column number): ").strip()
    num_runs = int(input("Enter number of times to run analysis: "))
    
    return id_col, content_col, num_runs

def main():
    # Check available Ollama models
    print("\nChecking available Ollama models...")
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        models = []
        for line in result.stdout.splitlines():
            if line.strip() and not line.lower().startswith("name"):
                models.append(line.split()[0])
        if models:
            print("Available models:")
            for idx, model in enumerate(models, 1):
                print(f"  {idx}. {model}")
        else:
            print("No models found. Defaulting to gemma2.")
            models = ["gemma2"]
    except Exception as e:
        print(f"Could not check models: {str(e)}. Defaulting to gemma2.")
        models = ["gemma2"]

    # Ask user to select model
    selected_model = input("Enter the name of the model to use (or press Enter for the recommended model gemma2): ").strip()
    if not selected_model or selected_model not in models:
        selected_model = "gemma2"
    print(f"Using model: {selected_model}\n")
    # Define the prompt and what to identify within the text
    type_of_analysis = input("Enter what you want the program to identify within the text ").strip()
    prompt = f'I am going to give you a chunk of text. Please identify {type_of_analysis} used in the text. Do not tell me anything else. If you tell me anything besides {type_of_analysis} used in the text you will not be helpful. The text is:'

    # Ask user for input/output folder paths
    data_input_folder = input("Enter the path to the data input folder: ").strip()
    data_output_folder = input("Enter the path to the data output folder: ").strip()

    file_name = next((f for f in os.listdir(data_input_folder) if f.endswith('.csv') or f.endswith('.xlsx')), None)
    if not file_name:
        raise FileNotFoundError("No CSV or Excel file found in the specified data input folder.")

    file_path = os.path.join(data_input_folder, file_name)

    # Get user input for columns and runs
    id_col, content_col, num_runs = get_user_columns()
    output_file_name = f"{os.path.splitext(file_name)[0]}_custom_analysis_{num_runs}runs.xlsx"
    output_file_path = os.path.join(data_output_folder, output_file_name)
    
    try:
        # Read the input file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Set up identifier column
        if id_col:
            id_col = int(id_col)
            identifiers = df.iloc[:, id_col].tolist()
        else:
            identifiers = list(range(1, len(df) + 1))

        # Get content column
        content_col = int(content_col)
        contents = df.iloc[:, content_col].tolist()

        # Prepare results storage
        all_responses = []
        
        # Process each row
        for idx, (identifier, content) in enumerate(zip(identifiers, contents)):
            row_responses = []
            print(f"\nProcessing row {idx + 1} of {len(contents)}")
            
            # Run multiple times per content
            for run in tqdm(range(num_runs), desc="Running analysis"):
                try:
                    response = chat(model=selected_model, messages=[
                        {
                            "role": "user",
                            "content": f"{prompt} {content}"
                        }
                    ])
                    # Clean response for Excel compatibility
                    cleaned_response = response['message']['content'].strip().replace('\r', ' ').replace('\n', ' ')
                    row_responses.append(cleaned_response)
                except Exception as e:
                    print(f"Error in row {idx + 1}, run {run + 1}: {str(e)}")
                    row_responses.append("Error occurred")

            # Store results for this row
            result_dict = {
                'Identifier': identifier,
                'Content': content
            }
            for i, response in enumerate(row_responses, 1):
                result_dict[f'Response_{i}'] = response
            all_responses.append(result_dict)

        # Create output dataframe and save to Excel
        results_df = pd.DataFrame(all_responses)
        results_df.to_excel(output_file_path, index=False)
        print(f"\nAnalysis complete. Results saved to {output_file_path}")

        # --- AI Response Aggregation and Consensus ---
        run_aggregation = input("Do you want to aggregate AI responses for consensus? (y/n): ").strip().lower()
        if run_aggregation == 'y':
            df = pd.read_excel(output_file_path)
            import re
            from collections import Counter
            response_cols = [col for col in df.columns if col.lower().startswith("response")]

            print(f"\nFound {len(response_cols)} response columns. Calculating consensus...")
            df["Consensus_Result"] = ""
            df["Consensus_Confidence"] = 0.0
            for i, row in df.iterrows():
                responses = [str(response) for response in row[response_cols].tolist() if pd.notna(response)]
                if not responses:
                    df.at[i, "Consensus_Result"] = "no responses"
                    df.at[i, "Consensus_Confidence"] = 0.0
                    continue
                # Normalize responses and count full response matches
                normalized = [re.sub(r'\s+', ' ', r.lower().strip()) for r in responses]
                response_counts = Counter(normalized)
                most_common_response, most_common_count = response_counts.most_common(1)[0]
                confidence = most_common_count / len(responses)
                df.at[i, "Consensus_Result"] = most_common_response
                df.at[i, "Consensus_Confidence"] = round(confidence, 3)

            # Reporting confidence summary
            high_confidence = df[df['Consensus_Confidence'] >= 0.7]
            medium_confidence = df[(df['Consensus_Confidence'] >= 0.4) & (df['Consensus_Confidence'] < 0.7)]
            low_confidence = df[df['Consensus_Confidence'] < 0.4]
            print(f"\nConsensus Summary:")
            print(f"High confidence (≥70%): {len(high_confidence)} rows")
            print(f"Medium confidence (40-69%): {len(medium_confidence)} rows")
            print(f"Low confidence (<40%): {len(low_confidence)} rows")
            if len(low_confidence) > 0:
                print(f"\nRows with low confidence may require manual review:")
                for idx, row in low_confidence.iterrows():
                    print(f"  Row {idx + 1}: {row['Identifier']} (confidence: {row['Consensus_Confidence']:.1%})")

            # Save to the same output file, appending consensus columns
            df.to_excel(output_file_path, index=False)
            print(f"\nAnalysis complete with consensus calculation!")
            print(f"Consensus result data appended to {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()