import pandas as pd
from tqdm import tqdm
from ollama import chat
from ollama import ChatResponse
import os
import argparse

def get_user_columns():
    """Get column selections from user input"""
    id_col = input("Enter the column number to use as identifier (or press Enter to use auto-numbering): ").strip()
    content_col = input("Enter the column number for content to analyze: ").strip()
    num_runs = int(input("Enter number of times to run analysis: "))
    
    return id_col, content_col, num_runs

def main():
    # Define the prompt and what to identify within the text
    type_of_analysis = input("Enter what you want the program to identify within the text ").strip()
    prompt = f'I am going to give you a chunk of text. Please identify {type_of_analysis} used in the text. Do not tell me anything else. If you tell me anything besides {type_of_analysis} used in the text you will not be helpful. The text is:'

    # Set up input/output paths
    data_input_folder = 'C:/Users/leves/Documents/GitHub/AI_Analysis_Tool/Data_Input'
    file_name = next((f for f in os.listdir(data_input_folder) if f.endswith('.csv') or f.endswith('.xlsx')), None)
    
    if not file_name:
        raise FileNotFoundError("No CSV or Excel file found in the Data_Input folder.")
        
    file_path = os.path.join(data_input_folder, file_name)
    output_file_name = os.path.splitext(file_name)[0] + '_custom_analysis.xlsx'
    output_file_path = os.path.join('C:/Users/leves/Documents/GitHub/AI_Analysis_Tool/Data_Output', output_file_name)

    # Get user input for columns and runs
    id_col, content_col, num_runs = get_user_columns()
    
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
                    response = chat(model="gemma2", messages=[
                        {
                            "role": "user",
                            "content": f"{prompt} {content}"
                        }
                    ])
                    row_responses.append(response['message']['content'])
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

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()