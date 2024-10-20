import os
import nltk

# Ensure NLTK word tokenizer is downloaded
nltk.download('punkt')

# Function to count the number of words in a text
def count_words(text):
    words = nltk.word_tokenize(text)
    return len(words)

# Function to read the content of a file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to process all text files in the directory and compute word counts
def compute_word_counts(directory):
    word_counts = []
    
    # Iterate over all files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(directory, file_name)
            text = read_file(file_path)
            word_count = count_words(text)
            word_counts.append(word_count)
    
    # Calculate average and maximum word count
    if word_counts:
        avg_word_count = sum(word_counts) / len(word_counts)
        max_word_count = max(word_counts)
        return avg_word_count, max_word_count
    else:
        return 0, 0  # If no text files are found, return zero

# Example: Process all text files in the directory
directory = '../../data/crawled/crawled_text_data_append_sharding_1024_200_max_1024'  # Specify the path to your text file directory
avg_word_count, max_word_count = compute_word_counts(directory)

print(f"Average Word Count: {avg_word_count}")
print(f"Maximum Word Count: {max_word_count}")