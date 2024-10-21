# This bash file is used to evaluate the performance of QA systems on the test set.
# example usage: bash evaluation/run_eval.sh

# valid command looks like this:
# python evaluation/eval.py --gold_answer_dir data/test/reference_answers.txt --generated_answer_dir data/test/generated_answers_{NAME_OF_YOUR_MODEL}.txt --output_dir results/{NAME_OF_YOUR_MODEL}_results.json

python evaluation/evaluate.py --combined_dir output/llama3_baseline.csv --output_dir results/llama3_baseline.json
# python evaluation/evaluate.py --combined_dir output/llama3_recursive_chroma_top3.csv --output_dir results/llama3_recursive_chroma_top3_results.json