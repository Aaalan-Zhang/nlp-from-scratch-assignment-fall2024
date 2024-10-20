'''
This file is for parsing static webpages. It is a part of the web crawling process.
Based on the manual collection of URLs, this script crawls the webpages and extracts ALL the text data.
'''

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import warnings
from urllib.parse import urljoin
from urllib3.exceptions import InsecureRequestWarning
import csv

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
def fetch_page_text(url, retries=3, timeout=5):
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
            time.sleep(1)  # Optional: wait before retrying

        if attempt == retries:
            return None, None

# Extract all sublinks (anchor tags) and handle relative URLs
def extract_sublinks(soup, base_url):
    sublinks = []
    for link in soup.find_all('a', href=True):
        sublink = link['href']
        # Convert relative URLs to absolute URLs using urljoin
        full_url = urljoin(base_url, sublink)
        sublinks.append(full_url)
    return sublinks

# Crawl through the URLs and fetch the data with retry
def crawl_urls(url_list):
    results = {}
    sublinks_data = []  # List to store (parent_url, sublink) pairs
    for index, url in enumerate(url_list):
        print(f"Fetching: {url}")
        text, soup = fetch_page_text(url)
        if text and soup:
            # Store the crawled text
            results[url] = text
            # Extract sublinks
            sublinks = extract_sublinks(soup, url)
            for sublink in sublinks:
                sublinks_data.append((url, sublink))
        else:
            print(f"Failed to parse URL at index: {index}, URL: {url}")
    return results, sublinks_data

# Save the crawled text data to .txt files
def save_crawled_data(crawled_data, urls):
    for index, url in enumerate(urls):
        # Fetch the text corresponding to the URL from the crawled_data dictionary
        text = crawled_data.get(url, "")

        # Remove all newline characters from the text
        cleaned_text = text.replace('\n', ' ')

        # Define the file name with the index from the URLs list
        output_file = f"../../data/crawled/crawled_text_data/{index}.txt"

        # Save the cleaned text to the file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

# Save sublinks to a CSV file
def save_sublinks_to_csv(sublinks_data, output_csv):
    df = pd.DataFrame(sublinks_data, columns=['Parent URL', 'Sublink'])
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    file_path = '../../data/raw/raw_csv_data/data_source.csv'
    data = pd.read_csv(file_path)

    # Extract non-empty URLs from the 'Source URL' column
    urls = data[data['Select'] == 'Webpage']['Source URL'].dropna().unique()

    # Start crawling the URLs
    crawled_data, sublinks_data = crawl_urls(urls)
    
    # Save the crawled text data
    save_crawled_data(crawled_data, urls)

    output_csv = '../../data/crawled/parentlink_file_name_url_mapping.csv'

    # Create the CSV mapping file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header (optional)
        writer.writerow(['Index', 'Value'])
        
        # Write the index and value pairs to the CSV
        for index, value in enumerate(urls):
            writer.writerow([index, value])

    print(f"CSV mapping file created at {output_csv}")
    # Save sublinks to a CSV file
    sublinks_output_file = "../../data/crawled/crawled_sublinks.csv"
    save_sublinks_to_csv(sublinks_data, sublinks_output_file)

    print("Crawling complete!")