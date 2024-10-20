'''
Script for pdf crawling and text extraction
'''

import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    # Create a PDF reader object
    reader = PdfReader(pdf_path)
    
    # Initialize a string to hold the extracted text
    extracted_text = ""
    
    # Loop through each page in the PDF
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text += page.extract_text() + "\n"  # Extract text from each page and add newline for separation
    
    return extracted_text

# Function to list all PDF files in a directory
def list_pdf_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.pdf')]

if __name__ == "__main__":
    # Directory containing the PDF files
    pdf_file_dir = '../../data/raw/raw_pdf_data'

    # Get all PDF files in the directory
    pdf_files = list_pdf_files(pdf_file_dir)

    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_file_path = os.path.join(pdf_file_dir, pdf_file)  # Full path to the PDF file
        print(f"Processing {pdf_file}...")

        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_file_path)

        # Create a corresponding text file name (same name as the PDF, but with .txt extension)
        txt_file_name = os.path.splitext(pdf_file)[0] + ".txt"
        txt_file_path = os.path.join('../../data/crawled/crawled_pdf_text_data', txt_file_name)

        # Save the extracted text to a text file
        with open(txt_file_path, 'w', encoding='utf-8') as text_file:
            cleaned_text = text.replace('\n', ' ')
            text_file.write(cleaned_text)

        print(f"Saved extracted text to {txt_file_name}")