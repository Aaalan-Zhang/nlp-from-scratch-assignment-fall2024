import pandas as pd
import re

def is_valid_url(url):
    pattern = re.compile(r'^https?://.*(?<!\.pdf)(?<!\.svg)(?<!\.png)(?<!\.jpg)$')
    return bool(pattern.match(url))

file_path = '/Users/alan/11711/nlp-from-scratch-assignment/data/crawled/sublink_file_name_url_mapping_checked_duplicate_sublink.csv'
data = pd.read_csv(file_path)

url_column = 'Value'

data_filtered = data[data[url_column].apply(is_valid_url)]

output_file = file_path.replace('_checked_duplicate_sublink.csv', '_cleaned_valid_url.csv')
data_filtered.to_csv(output_file, index=False)

print(f"filtering is complete and the files has been saved to {output_file}")