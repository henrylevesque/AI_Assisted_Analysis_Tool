import pandas as pd
from tqdm import tqdm
from ollama import chat
from ollama import ChatResponse
import os
import argparse
import re
from collections import Counter

def get_user_columns():
    """Get column selections from user input"""
    id_col = input("Enter the column number to use as identifier (or press Enter to use auto-numbering): ").strip()
    content_col = input("Enter the column number for content to analyze: ").strip()
    num_runs = int(input("Enter number of times to run analysis: "))
    
    return id_col, content_col, num_runs

def calculate_consensus(df, response_cols):
    """Calculate consensus from multiple AI responses"""
    print("\nCalculating consensus from AI responses...")
    
    # Create new columns for consensus results
    df["Consensus_Result"] = ""
    df["Consensus_Confidence"] = 0.0
    
    # Calculate consensus for each row
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing consensus"):
        responses = [str(response) for response in row[response_cols].tolist() if pd.notna(response)]
        
        if not responses:
            df.at[i, "Consensus_Result"] = "no responses"
            df.at[i, "Consensus_Confidence"] = 0.0
            continue
        
        # Handle single response case
        if len(responses) == 1:
            df.at[i, "Consensus_Result"] = responses[0]
            df.at[i, "Consensus_Confidence"] = 1.0
            continue
        
        # Normalize the responses to lowercase and split into words
        normalized_responses = [re.split(r'\s+', str(r).lower().strip()) for r in responses]
        
        # Flatten the list of lists and count occurrences of each word
        all_words = [word for response in normalized_responses for word in response if word]
        
        if not all_words:
            df.at[i, "Consensus_Result"] = "no valid responses"
            df.at[i, "Consensus_Confidence"] = 0.0
            continue
        
        word_counts = Counter(all_words)
        total_words = len(set(all_words))
        
        # Find words that appear in more than one response
        consensus_words = [word for word, count in word_counts.items() if count > 1]
        
        if consensus_words:
            # Sort by frequency and join to form consensus result
            consensus_words_sorted = sorted(consensus_words, key=lambda x: word_counts[x], reverse=True)
            consensus_result = ' '.join(consensus_words_sorted)
            confidence = len(consensus_words) / total_words if total_words > 0 else 0
            
            df.at[i, "Consensus_Result"] = consensus_result
            df.at[i, "Consensus_Confidence"] = round(confidence, 3)
        else:
            # No consensus found - use the most common response
            response_counts = Counter(responses)
            most_common_response = response_counts.most_common(1)[0][0]
            confidence = response_counts[most_common_response] / len(responses)
            
            df.at[i, "Consensus_Result"] = most_common_response
            df.at[i, "Consensus_Confidence"] = round(confidence, 3)
    
    return df

def run_custom_analysis():
    """Run the custom AI analysis"""
    print("=== AI-Assisted Custom Analysis Tool ===\n")
    
    # Define the prompt and what to identify within the text
    type_of_analysis = input("Enter what you want the program to identify within the text: ").strip()
    prompt = f'I am going to give you a chunk of text. Please identify {type_of_analysis} used in the text. Do not tell me anything else. If you tell me anything besides {type_of_analysis} used in the text you will not be helpful. The text is:'

    # Set up input/output paths - use relative paths to work in any environment
    data_input_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data_Input')
    data_output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data_Output')
    
    # Create output folder if it doesn't exist
    os.makedirs(data_output_folder, exist_ok=True)
    
    # Find input file
    if not os.path.exists(data_input_folder):
        print(f"Error: Data_Input folder not found at {data_input_folder}")
        print("Please create a Data_Input folder in the main project directory and add your CSV/Excel file.")
        return None, None
    
    file_name = next((f for f in os.listdir(data_input_folder) if f.endswith('.csv') or f.endswith('.xlsx')), None)
    
    if not file_name:
        print(f"No CSV or Excel file found in {data_input_folder}")
        print("Please add a CSV or Excel file to the Data_Input folder.")
        return None, None
        
    file_path = os.path.join(data_input_folder, file_name)
    
    # Create output filename
    base_name = os.path.splitext(file_name)[0]
    output_file_name = f'{base_name}_custom_analysis_with_consensus.xlsx'
    output_file_path = os.path.join(data_output_folder, output_file_name)

    # Get user input for columns and runs
    id_col, content_col, num_runs = get_user_columns()
    
    try:
        # Read the input file
        print(f"\nReading input file: {file_name}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        print(f"Found {len(df)} rows to process")
        print(f"Columns available: {list(df.columns)}")

        # Set up identifier column
        if id_col:
            id_col = int(id_col)
            identifiers = df.iloc[:, id_col].tolist()
            print(f"Using column {id_col} as identifier")
        else:
            identifiers = list(range(1, len(df) + 1))
            print("Using auto-numbering as identifier")

        # Get content column
        content_col = int(content_col)
        contents = df.iloc[:, content_col].tolist()
        print(f"Using column {content_col} for content analysis")

        # Prepare results storage
        all_responses = []
        
        print(f"\nStarting analysis with {num_runs} runs per row...")
        
        # Process each row
        for idx, (identifier, content) in enumerate(zip(identifiers, contents)):
            row_responses = []
            print(f"\nProcessing row {idx + 1} of {len(contents)}")
            
            # Run multiple times per content
            for run in tqdm(range(num_runs), desc=f"Row {idx+1} analysis"):
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

        # Create output dataframe
        results_df = pd.DataFrame(all_responses)
        
        return results_df, output_file_path

    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
        return None, None

def main():
    """Main function that runs analysis and aggregation"""
    # Run the custom analysis
    results_df, output_file_path = run_custom_analysis()
    
    if results_df is None:
        print("Analysis failed. Exiting.")
        return
    
    # Find response columns for consensus calculation
    response_cols = [col for col in results_df.columns if col.lower().startswith("response")]
    
    if len(response_cols) > 1:
        print(f"\nFound {len(response_cols)} response columns. Calculating consensus...")
        
        # Calculate consensus
        results_df = calculate_consensus(results_df, response_cols)
        
        # Save results with consensus
        results_df.to_excel(output_file_path, index=False)
        print(f"\nAnalysis complete with consensus calculation!")
        print(f"Results saved to: {output_file_path}")
        
        # Show consensus summary
        high_confidence = results_df[results_df['Consensus_Confidence'] >= 0.7]
        medium_confidence = results_df[(results_df['Consensus_Confidence'] >= 0.4) & (results_df['Consensus_Confidence'] < 0.7)]
        low_confidence = results_df[results_df['Consensus_Confidence'] < 0.4]
        
        print(f"\nConsensus Summary:")
        print(f"High confidence (≥70%): {len(high_confidence)} rows")
        print(f"Medium confidence (40-69%): {len(medium_confidence)} rows")
        print(f"Low confidence (<40%): {len(low_confidence)} rows")
        
        if len(low_confidence) > 0:
            print(f"\nRows with low confidence may require manual review:")
            for idx, row in low_confidence.iterrows():
                print(f"  Row {idx + 1}: {row['Identifier']} (confidence: {row['Consensus_Confidence']:.1%})")
        
    else:
        print(f"\nOnly one response column found. Saving without consensus calculation.")
        results_df.to_excel(output_file_path, index=False)
        print(f"Results saved to: {output_file_path}")

if __name__ == "__main__":
    main()
