import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

# Make sure NLTK's sentence and word tokenizers are downloaded
nltk.download('punkt')

MAX_WORDS_PER_CHUNK = 3072  # Maximum number of words per shard
OVERLAP_WORDS = 600  # Number of overlapping words between shards
# Function to read the content of a text file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to split text into sentence-based chunks with word-based overlap
def shard_text_by_sentences(text, max_words_per_chunk=MAX_WORDS_PER_CHUNK, overlap_words=OVERLAP_WORDS):
    sentences = sent_tokenize(text)  # Tokenize text into sentences
    chunks = []
    current_chunk = []
    current_word_count = 0

    for i, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence)
        sentence_word_count = len(sentence_words)

        # Check if adding this sentence will exceed the word limit for the chunk
        if current_word_count + sentence_word_count > max_words_per_chunk:
            # Finalize the current chunk
            chunks.append(current_chunk)

            # Reset current_chunk with overlap from the previous chunk and new sentence
            overlap = current_chunk[-overlap_words:] if len(current_chunk) > overlap_words else current_chunk
            current_chunk = overlap + sentence_words
            current_word_count = len(current_chunk)

        else:
            current_chunk += sentence_words
            current_word_count += sentence_word_count

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Function to save each chunk into a separate text file
def save_shard(chunk, output_dir, file_name, shard_index):
    output_file = os.path.join(output_dir, f"{file_name}-{shard_index}.txt")
    
    # Join words in the chunk and save to file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(' '.join(chunk))

# Function to process all text files in a directory
def process_directory(input_dir, output_dir, max_words_per_chunk=1024, overlap_words=200):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_dir, file_name)
            file_base_name = os.path.splitext(file_name)[0]

            # Read file content and shard it
            text = read_file(file_path)
            shards = shard_text_by_sentences(text, max_words_per_chunk, overlap_words)

            # Save each shard
            for index, shard in enumerate(shards):
                save_shard(shard, output_dir, file_base_name, index)

    print("All files processed.")

# Define input and output directories
input_directory = '../../data/crawled/crawled_text_data'  # Directory containing text files
output_directory = f'../../data/crawled/crawled_text_data_append_sharding_{MAX_WORDS_PER_CHUNK}_{OVERLAP_WORDS}'  # Directory to save the shards

# Process the directory and shard files
process_directory(input_directory, output_directory)