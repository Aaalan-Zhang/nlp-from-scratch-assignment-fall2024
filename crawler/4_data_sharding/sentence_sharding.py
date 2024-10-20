import os
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Make sure NLTK's sentence tokenizer is downloaded
nltk.download('punkt_tab')
nltk.download('punkt')

CHUNK_SIZE = 50
OVERLAP = 5
# Read the content of a file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Split text into overlapping sentence chunks
def shard_sentences(text, chunk_size=CHUNK_SIZE, overlap_prev=OVERLAP, overlap_next=OVERLAP):
    sentences = sent_tokenize(text)  # Tokenize text into sentences
    step_size = chunk_size - overlap_prev  # Step size for sliding window
    for i in range(0, len(sentences), step_size):
        start = max(0, i - overlap_prev)  # Ensure we include the previous overlap
        end = min(len(sentences), i + chunk_size + overlap_next)
        yield sentences[start:end]

# Save each shard into a separate file
def save_shard(shard, output_dir, file_name, shard_index):
    output_file = os.path.join(output_dir, f"{file_name}-{shard_index}.txt")
    
    # Save the shard into the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(' '.join(shard))

# Process each text file in the directory
def process_directory(input_dir, output_dir, chunk_size=CHUNK_SIZE, overlap_prev=OVERLAP, overlap_next=OVERLAP):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over all text files in the input directory
    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(input_dir, file_name)
            file_base_name = os.path.splitext(file_name)[0]  # Remove file extension

            # Read the file content and shard it
            text = read_file(file_path)
            shards = list(shard_sentences(text, chunk_size, overlap_prev, overlap_next))
            
            # Save each shard
            for index, shard in enumerate(shards):
                save_shard(shard, output_dir, file_base_name, index)

    print("All files processed.")

input_directory = '../../data/crawled/crawled_text_data'  # The directory containing your text files
output_directory = f'../../data/crawled/crawled_text_data_sentence_{CHUNK_SIZE}_{OVERLAP}'  # The directory to save the output files
os.makedirs(output_directory, exist_ok=True)
process_directory(input_directory, output_directory)