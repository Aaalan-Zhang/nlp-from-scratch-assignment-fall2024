import os
from transformers import pipeline
from tqdm import tqdm

# Load the pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Read the content of a file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Summarize a text chunk
def summarize_text(text, max_length=50):
    return summarizer(text, max_length=max_length, min_length=25, do_sample=False)[0]['summary_text']

# Process each shard file in the directory and create summaries
def process_directory(input_dir, output_dir, summary_output_dir, max_summary_length=50):
    # Ensure the output directories exist
    if not os.path.exists(summary_output_dir):
        os.makedirs(summary_output_dir)

    # Iterate over all text files (shards) in the input directory
    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(input_dir, file_name)
            text = read_file(file_path)
            
            # Generate the summary for the current shard
            summary = summarize_text(text, max_length=max_summary_length)
            
            # Save the summary to a new file
            summary_file_path = os.path.join(summary_output_dir, f"{file_name}_summary.txt")
            with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                summary_file.write(summary)

            print(f"Summary for {file_name} saved to {summary_file_path}")

# Example: Process all shards in the input directory and generate summaries
input_directory = '/Users/alan/11711/nlp-from-scratch-assignment/data/crawled/crawled_text_data_test'  # Directory with text shards
summary_output_directory = '/Users/alan/11711/nlp-from-scratch-assignment/data/crawled/output_summary'  # Directory to save summaries

process_directory(input_directory, summary_output_directory, summary_output_directory)