### baseline performance
```
python evaluation/evaluate.py --combined_dir output/llama3_baseline.csv --output_dir results/llama3_baseline.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chroma_top3.csv --output_dir results/llama3_recursive_chroma_top3.json
```

### for hyperparameter tuning on chunk size
```
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk2000_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk2000_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk500_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk500_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk750_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk750_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_recursive_chunk1500_chroma_top3_sample100.csv --output_dir results/llama3_recursive_chunk1500_chroma_top3_sample100.json
```

### for hyperparameter tuning on splitter
```
python evaluation/evaluate.py --combined_dir output/llama3_character_chroma_top3_sample100.csv --output_dir results/llama3_character_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_tokensplit_chroma_top3_sample100.csv --output_dir results/llama3_tokensplit_chroma_top3_sample100.json
python evaluation/evaluate.py --combined_dir output/llama3_semantic_chroma_top3_sample100.csv --output_dir results/llama3_semantic_chroma_top3_sample100.json
```

### for tuning reranking using faiss
```
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test.csv --output_dir results/llama3_faiss_test.json
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test_rerank.csv --output_dir results/llama3_faiss_test_rerank.json
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test_rerank_t5.csv --output_dir results/llama3_faiss_test_rerank_t5.json
python evaluation/evaluate.py --combined_dir output/llama3_faiss_test_rerank_MiniLM.csv --output_dir results/llama3_faiss_test_rerank_MiniLM.json
```