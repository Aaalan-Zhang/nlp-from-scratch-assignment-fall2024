# Data Directory

The data directory stores the data needed for the RAG.

The subdiretory `./raw` contains a list of manually collected URLs (as a CSV file named `data_source.csv` in the folder `raw_csv_data`). There are 161 URLs in the CSV, 9 of them are PDF files which will require PDF parsing. The original PDF files are in the folder `raw_pdf_data` .

The subdirectory `./crawled` contains these subdirectories/files:

1. `crawled_pdf_text_data`
2. `crawled_text_data`
3. `crawled_all` (in the format of `crawled_all.txt`)
4. `parentlink_file_name_url_mapping.csv`
5. `crawled_sublinks_original.csv`
6. `sublink_file_name_url_mapping.csv`
7. `sublink_file_name_url_mapping_filtered.csv`

`crawled_pdf_text_data` gives the parsed text data for each PDF file. The file name is identical to the original PDF file.

`crawled_text_data` gives the crawled text data for each URL in the file `data_source.csv` in the raw data directory. The file names are numbers. The mapping between the numbers and the actual URL is given in the file `parentlink_file_name_url_mapping.csv`.

`crawled_sublinks_original.csv` gives the mapping between parent link and the sublink. Note that it may have a lot of duplicates and URLs which represent the same webpage.

`crawled_all` gives the crawled text data for each parentlink URL, sublink URL, and parsed PDF texts combined in the file `crawled_sublinks_original.csv` in the `.data/crawled` directory and `data_source.csv` in the `.data/raw` directory. The file names are file source + index numbers. A file can be named `parentlink_{index}.txt` or `sublink_{index}.txt` or `pdf_{pdf_name}.txt` based on the crawled URL type.

To find the actual URL crawled for that file, you need to look up the `sublink_file_name_url_mapping.csv` if the text file name begins with `sublink`, `parentlink_file_name_url_mapping.csv` if the text file name begins with `parentlink`, or the `.data/raw/raw_pdf_data` for the actual PDF file.

In the file  `sublink_file_name_url_mapping.csv`, the column `Index_file1` represents the `{index}` in the file name. For example, if you want to find out what URL is for `sublink_255.txt`, look up 255 in that column and you will find `https://en.wikipedia.org/wiki/Flag_of_Pittsburgh`. For the file `parentlink_file_name_url_mapping.csv`, check out the column `Index` for the `{index}` in the file name.

Note that because `crawled_all` contains 13001 files, the original folder size is about 270MB. I decided to compress it. So, a `crawled_all.txt` is replaced here, which contains a Google Drive URL for the download link of this compressed file. The compressed size is about 100MB.

## Maintainer

Alan Zhang (chengliz)
