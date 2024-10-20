'''
Dynamic crawling for the events data
'''

import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm

# Function to fetch the page content using Selenium
def fetch_page_text_selenium(url):
    try:
        driver = webdriver.Chrome(driverPath) 
        # Initialize Chrome WebDriver
        driver.get(url)
        
        # Wait for the page to load (adjust time if needed)
        time.sleep(3)
        
        # Get page source and close the browser
        page_source = driver.page_source
        driver.quit()

        # Parse the page content using BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract and clean the text from the page
        page_text = soup.get_text(separator='\n', strip=True)
        return page_text

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

# Function to read URLs from CSV and crawl each one
def crawl_urls_from_csv(csv_file_path, url_column_name):
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            url = row[url_column_name]
            print(f"Fetching: {url}")
            text = fetch_page_text_selenium(url)
            if text:
                # Save the crawled text to a file with the index as the filename
                output_file = f"../../data/crawled/events_test/{index}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Saved content to {output_file}")

if __name__ == "__main__":
    csv_file_path = '../../data/raw/raw_csv_data/events_after_10_27.csv'
    url_column_name = 'Source URL'
    driverPath = 'chromedriver'


    # Start crawling the URLs from the CSV
    crawl_urls_from_csv(csv_file_path, url_column_name)