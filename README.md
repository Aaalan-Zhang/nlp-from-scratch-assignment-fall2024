# NLP From Scratch

This is the project repo for 11-711 ANLP Fall 24 Project 2. The specs can be found in the file Info.md.

## How to use

First, it is recommended that you have a machine with GPU with memory > 20GB, CUDA support, and at least 50GB available disk memory. Then, install all the required packages (make sure you are in the root directory of this repo).

```
pip install -r requirements
```

To run the RAG pipeline, in the root directory, execute

```
python pipeline/rag_pipeline_new.py \
--model_name meta-llama/Llama-3.1-8B-Instruct \
--dtype float16 \
--embedding_model_name sentence-transformers/all-MiniLM-L6-v2 \
--embedding_dim 384 \ 
--splitter_type recursive \
--chunk_size 1000 \
--chunk_overlap 200 \
--text_files_path data/crawled/crawled_text_data \
--top_k_search 3 \
--retriever_type FAISS \
--rerank_model_name ms-marco-MiniLM-L-12-v2 \
--hypo False \
--output_file output/baseline_rag.csv
```

Please see `.pipeline/rag_pipeline_new.py` for a list of full available argument options.

To run the evaluation, in the root directory, execute

```
python evaluation/evaluate.py --combined_dir output/baseline_rag.csv --output_dir results/baseline_rag.json
```

## Members

The team members are (ordered by name, last name, then first name):

Haojun Liu (haojunli)

Qingyang Liu (qliu3)

Chenglin Zhang (chengliz)
