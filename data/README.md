# Data Directory

The data directory stores the data needed for the RAG. 

The subdiretory `./raw` contains a list of manually collected URLs (as a CSV file named `data_source.csv` in the folder `raw_csv_data`). There are 161 URLs in the CSV, 9 of them are PDF files which will require PDF parsing. The original PDF files are in the folder `raw_pdf_data` .

The subdirectory `./crawled` contains these subdirectories/files:

1. `crawled_pdf_text_data`
2. `crawled_text_data`
3. `crawled_sublink_text_data` (in the format of `crawled_sublink_text_data.txt`)
4. `parentlink_file_name_url_mapping.csv`
5. `crawled_sublinks_original.csv`
6. `sublink_file_name_url_mapping.csv`
7. `sublink_file_name_url_mapping_cleaned.csv`

`crawled_pdf_text_data` gives the parsed text data for each PDF file. The file name is identical to the original PDF file.

`crawled_text_data` gives the crawled text data for each URL in the file `data_source.csv` in the raw data directory. The file names are numbers. The mapping between the numbers and the actual URL is given in the file `parentlink_file_name_url_mapping.csv`.

`crawled_sublinks_original.csv` gives the mapping between parent link and the sublink. Note that it may have a lot of duplicates and URLs which represent the same webpage.

`crawled_sublink_text_data` gives the crawled text data for each sublink URL in the file `crawled_sublinks_original.csv`. The file names are numbers. The mapping between the numbers and the actual URL is given in the file `sublink_file_name_url_mapping.csv`. Note that because it contains over 18000 files, even if the folder is compressed, it is still larger than the file size permitted to be uploaded to GitHub. So, a `crawled_sublink_text_data.txt` is replaced here, which contains a Google Drive URL for the download link of this compressed file. The size is about 200MB.

Because `sublink_file_name_url_mapping.csv` contains URLs which represent the same webpage (for example, `https://en.wikipedia.org/wiki/Pittsburgh#Images`, `https://en.wikipedia.org/wiki/Pittsburgh#Climate`, and `https://en.wikipedia.org/wiki/Pittsburgh`), `sublink_file_name_url_mapping_cleaned.csv` further drops the duplicates and gives a mapping of the file names in `crawled_sublink_text_data` and their represented URLs.

## Maintainer

Alan Zhang (chengliz)
