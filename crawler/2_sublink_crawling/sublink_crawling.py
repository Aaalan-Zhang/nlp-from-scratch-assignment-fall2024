'''
This file is for parsing static webpages from the collected sublink file. It is a part of the web crawling process.
Based on the URLs of the collected sublink URLs, this script crawls the webpages and extracts ALL the text data (if opening the sublink URL does not timeout).
'''

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import warnings
from urllib.parse import urljoin
from urllib3.exceptions import InsecureRequestWarning
from tqdm import tqdm
import csv

# Define user agents
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
]
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# Define headers for requests
headers = {'User-Agent': random.choice(user_agents)}
session = requests.Session()

# Fetch the page text with retry mechanism
def fetch_page_text(url, retries=1, timeout=5):
    attempt = 0
    while attempt < retries:
        try:
            # Send a GET request to the URL with a timeout
            response = session.get(url, timeout=timeout, headers=headers, verify=False)
            response.raise_for_status()  # Raise an error for bad responses
            
            # Parse the page content
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract and return the text from the page and the soup object
            return soup.get_text(separator='\n', strip=True), soup
        
        except requests.exceptions.RequestException as e:
            attempt += 1

        if attempt == retries:
            return None, None

# Save the crawled text data to .txt files
def save_crawled_text(url, text, index):
    # Clean up the text by removing newlines
    cleaned_text = text.replace('\n', ' ')

    # Define the file name with the index from the URLs list
    output_file = f"../../data/crawled/crawled_sublink_text_data/{index}.txt"

    # Save the cleaned text to the file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

# Crawl through the URLs, parse and save one by one
def crawl_and_save(url_list, indexes):
    for index, url in tqdm(enumerate(url_list), total=len(url_list)):
        if url.startswith('http'):
            text, _ = fetch_page_text(url)
            if text:
                # Save the parsed text immediately after parsing
                save_crawled_text(url, text, indexes[index])
            else:
                print(f"Failed to parse URL at index {index}, URL: {url}")

if __name__ == "__main__":
    # Read the CSV file with URLs
    file_path = '../../data/crawled/sublink_file_name_url_mapping_filtered.csv'
    data = pd.read_csv(file_path)

    # Extract non-empty URLs from the 'Source URL' column
    urls = data['Value']
    indexes = data['Index_file1']

    # Start crawling the URLs and save the result one by one
    crawl_and_save(urls, indexes)

    print("Crawling complete!")