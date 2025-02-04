import pandas as pd
from tqdm import tqdm
from ollama import chat
from ollama import ChatResponse
import os

# Define the initial content as a variable
initial_user_content = 'I am going to give you the text of an abstract. Please identify the methods used in the abstract. Do not tell me anything else. If you tell me anything besides the methods used in the abstract you will not be helpful. The abstract text is:'

# Read the content from a specific cell in a CSV or Excel file
data_input_folder = 'C:/Users/leves/Documents/GitHub/AI_Assisted_Analysis/Data_Input'
file_name = next((f for f in os.listdir(data_input_folder) if f.endswith('.csv') or f.endswith('.xlsx')), None)
if file_name:
    file_path = os.path.join(data_input_folder, file_name)
else:
    raise FileNotFoundError("No CSV or Excel file found in the Data_Input folder.")
sheet_name = 'Sheet1'  # Only needed for Excel files
start_row = 0  # Starting row index (0-based)
title_col = 4  # Column index for title (0-based)
abstract_col = 10  # Column index for abstract (0-based)
author_col = 3  # Column index for author name(s) (0-based)
publication_date_col = 2  # Column index for publication date (0-based)

# Output file path
output_file_name = os.path.splitext(file_name)[0] + '_methods_ai_responses.xlsx'
output_file_path = os.path.join('C:/Users/leves/Documents/GitHub/AI_Assisted_Analysis/Data_Output', output_file_name)

# Number of times to run the loop
num_runs = 5

try:
    # For CSV files
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    # For Excel files
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    # Initialize the responses list with publication date, author, titles, and abstracts
    responses = [[df.iloc[row, publication_date_col], df.iloc[row, author_col], df.iloc[row, title_col], df.iloc[row, abstract_col]] for row in range(start_row, len(df))]

    # Loop through each run
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        # Loop through each row starting from start_row with a progress bar
        for row in tqdm(range(start_row, len(df)), desc=f"Processing rows for run {run + 1}"):
            title = df.iloc[row, title_col]
            abstract = df.iloc[row, abstract_col]
            user_content = initial_user_content + f' {abstract}'

            response: ChatResponse = chat(model='gemma2', messages=[
              {
                'role': 'user',
                'content': user_content,
              },
            ])
            response_content = response.message.content if isinstance(response, ChatResponse) else "Error: Invalid response type"
            
            # Append the response to the corresponding row in the responses list
            responses[row - start_row].append(response_content)

    # Ensure each row in responses has the correct number of elements
    for row in responses:
        while len(row) < 4 + num_runs:
            row.append("Error: No response")

    # Create a DataFrame from the responses
    columns = ['Publication Date', 'Author', 'Title', 'Abstract'] + [f'Response {i+1}' for i in range(num_runs)]
    response_df = pd.DataFrame(responses, columns=columns)

    # Write the DataFrame to an Excel file
    response_df.to_excel(output_file_path, index=False, engine='openpyxl')

    print(f"Responses have been written to {output_file_path}")

except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except pd.errors.ParserError:
    print("Error: There was a problem parsing the file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")