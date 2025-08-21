import os
import pandas as pd
from tqdm import tqdm
import re
from collections import Counter

def main():
    # Read all Excel files from Data_Output folder
    input_folder = "Data_Input"
    excel_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.xlsx')]

    if not excel_files:
        print("No Excel files found in the Data_Input folder.")
        return

    all_dfs = []
    for excel_file in excel_files:
        df = pd.read_excel(excel_file)
        all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    # Find all columns labeled "Response" (case insensitive and allowing text after "Response")
    response_cols = [col for col in df.columns if col.lower().startswith("response")]

    # Create new columns
    df["Consensus Result"] = ""
    df["Consensus over Total"] = 0.0

    # Calculate consensus
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        responses = row[response_cols].tolist()
        if not responses:
            df.at[i, "Consensus Result"] = "no consensus"
            df.at[i, "Consensus over Total"] = 0
            continue
        # Normalize the responses to lowercase and split into words
        normalized_responses = [re.split(r'\s+', r.lower()) for r in responses]
        # Flatten the list of lists and count occurrences of each word
        word_counts = Counter(word for response in normalized_responses for word in response)
        # Find the most common words
        most_common_words = [word for word, count in word_counts.items() if count > 1]
        if most_common_words:
            # Join the most common words to form the consensus result
            consensus_result = ' '.join(most_common_words)
            df.at[i, "Consensus Result"] = consensus_result
            df.at[i, "Consensus over Total"] = len(most_common_words) / len(word_counts)
        else:
            df.at[i, "Consensus Result"] = "no consensus"
            df.at[i, "Consensus over Total"] = 0

    # Save to a new Excel file in Data_Output folder
    output_folder = "Data_Output"
    os.makedirs(output_folder, exist_ok=True)
    for excel_file in excel_files:
        output_file_name = os.path.splitext(os.path.basename(excel_file))[0] + "_aggregated_data.xlsx"
        output_file = os.path.join(output_folder, output_file_name)
        df.to_excel(output_file, index=False)
        print(f"Consensus result data saved to {output_file}")

if __name__ == "__main__":
    main()
