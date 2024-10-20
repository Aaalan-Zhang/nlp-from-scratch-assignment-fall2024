import pandas as pd
from urllib.parse import urlparse, urlunparse

# Function to normalize a URL by removing the fragment (i.e., anything after #)
def normalize_url(url):
    parsed_url = urlparse(url)
    # Reconstruct the URL without the fragment
    return urlunparse(parsed_url._replace(fragment=''))

# Function to check and remove duplicate URLs based on the base URL (without fragments)
def remove_duplicate_urls(file_path, column_name):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Normalize the URLs (i.e., remove the fragment part)
    data['normalized_url'] = data[column_name].apply(normalize_url)

    # Remove rows with missing (NaN) URLs
    data = data.dropna(subset=['normalized_url'])

    # Remove duplicate rows based on the 'normalized_url' column while keeping the first occurrence
    data_cleaned = data.drop_duplicates(subset='normalized_url', keep='first')

    # Keep the original index from the input data
    data_cleaned = data_cleaned.reset_index(drop=False)  # Keep the original index in a new column 'index'

    # Add a new index column starting from 0
    data_cleaned['new_index'] = range(len(data_cleaned))

    # Drop the 'normalized_url' column as it's no longer needed
    data_cleaned = data_cleaned.drop(columns=['normalized_url'])

    # Save the cleaned data back to a CSV file
    output_file = file_path.replace('.csv', '_checked_duplicate_sublink.csv')
    data_cleaned.to_csv(output_file, index=False)

    print(f"Duplicates removed. Cleaned data saved to {output_file}")

if __name__ == "__main__":
    # Path to the CSV file
    file_path = '/Users/alan/11711/nlp-from-scratch-assignment/data/1010_160_entries/crawled/sublink_file_name_url_mapping.csv'
    
    # Column in the CSV file containing the URLs
    column_name = 'Value'  # Adjust based on your CSV column name

    # Remove duplicates
    remove_duplicate_urls(file_path, column_name)