import os
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Make sure NLTK's word tokenizer is downloaded
import nltk
nltk.download('punkt')

# Function to read the content of a text file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to split a large chunk into smaller chunks if it exceeds 1024 words
def split_chunk_by_words(text, max_words_per_chunk=1024):
    words = word_tokenize(text)  # Tokenize the chunk into words
    chunks = []
    
    for i in range(0, len(words), max_words_per_chunk):
        chunk = words[i:i + max_words_per_chunk]
        chunks.append(chunk)

    return chunks

# Function to save the new split chunks into separate text files
def save_shard(chunk, output_dir, file_name, shard_index):
    output_file = os.path.join(output_dir, f"{file_name}-{shard_index}.txt")
    
    # Join words in the chunk and save to file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(' '.join(chunk))

# Function to process the directory of chunks and further split chunks larger than 1024 words
def process_chunk_directory(input_dir, output_dir, max_words_per_chunk=1024):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_dir, file_name)
            file_base_name = os.path.splitext(file_name)[0]

            # Read the chunk file content
            text = read_file(file_path)
            
            # Check the word count of the chunk and split if necessary
            chunks = split_chunk_by_words(text, max_words_per_chunk)

            # Save the new smaller chunks
            for index, chunk in enumerate(chunks):
                save_shard(chunk, output_dir, file_base_name, index)

    print("All files processed.")

# Define input and output directories
input_directory = '../../data/crawled/crawled_text_data_append_sharding_1024_200'  # Directory containing text files
output_directory = '../../data/crawled/crawled_text_data_append_sharding_1024_200_max_1024'  # Directory to save the shards

os.makedirs(output_directory, exist_ok=True)
# Process the directory and split chunks if needed
process_chunk_directory(input_directory, output_directory)