# This bash file is used to evaluate the performance of QA systems on the test set.
# example usage: bash evaluation/run_eval.sh

# python evaluation/evaluate.py --combined_dir output/llama3_baseline.csv --output_dir results/llama3_baseline.json
# python evaluation/evaluate.py --combined_dir output/llama3_recursive_chroma_top3.csv --output_dir results/llama3_recursive_chroma_top3.json

# # for hyperparameter tuning
# python evaluation/evaluate.py --combined_dir output/llama3_recursive_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chroma_top3_sample100.json
# python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk2000_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk2000_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk500_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk500_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk750_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk750_chroma_top3_sample100.json