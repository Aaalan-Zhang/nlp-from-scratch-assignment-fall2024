import os
import torch
import pandas as pd
from tqdm import tqdm

import faiss
import numpy as np
import argparse

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain import hub
from langchain_chroma import Chroma
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, 
    CharacterTextSplitter, 
    TokenTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    PromptTemplate
)
from sentence_transformers import SentenceTransformer

from utility.rag_utility import (
    FAISSRetriever, 
    SentenceTransformerEmbeddings, 
    load_text_files,  
    answer_generation, 
    PROMPT_TEMPLATE
)

# ========================================
# Vars that can be set and read from another var file.
# ========================================


def parse_args():
    parser = argparse.ArgumentParser(description="Script for running RAG pipeline with FAISS or CHROMA.")

    # Add arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Name of the Hugging Face model to use.")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Precision type (float16 or bfloat16).")
    parser.add_argument("--embedding_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Name of the embedding model to use.")
    parser.add_argument("--embedding_dim", type=int, default=384, help="Dimension of the embeddings.")
    parser.add_argument("--splitter_type", type=str, choices=["recursive", "character", "token", "semantic"], default="Recursive",
                        help="Type of text splitter to use.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of the text chunks.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between text chunks.")
    parser.add_argument("--text_files_path", type=str, default="data/crawled/crawled_text_data",
                        help="Path to the text files directory.")
    parser.add_argument("--top_k_search", type=int, default=3, help="Top K documents to retrieve.")
    parser.add_argument("--retriever_type", type=str, choices=["FAISS", "CHROMA"], default="FAISS",
                        help="Type of retriever to use (FAISS or CHROMA).")
    parser.add_argument("--qes_file_path", type=str, default="data/annotated/QA_pairs_1.csv",
                        help="Path to the QA file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")

    return parser.parse_args()

# ========================================
# Main Script Execution
# ========================================
if __name__ == "__main__":

    # Step 0: Load environment variables
    load_dotenv()

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "rag-template"
    os.environ["USER_AGENT"] = "LangChain/1.0 (+https://www.langchain.com)"

    login(token=os.getenv('HUGGINGFACE_TOKEN'))

    args = parse_args()

    # Set model name, precision, and other parameters based on passed args
    model_name = args.model_name
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    embedding_model_name = args.embedding_model_name
    embedding_dim = args.embedding_dim
    splitter_type = args.splitter_type
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    text_files_path = args.text_files_path
    qes_file_path = args.qes_file_path
    top_k_search = args.top_k_search
    retriever_type = args.retriever_type
    output_file = args.output_file

    # Step 1: Initialize the Hugging Face model as your LLM
    print("Initializing the Hugging Face model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer, 
        torch_dtype=dtype
    )
    print("Model initialized successfully!")

    # Step 2: Load the Sentence Transformers model for embeddings
    embedding_model = SentenceTransformer(embedding_model_name, truncate_dim=embedding_dim)

    # Step 3: load the text files for building the index and qa evaluation
    print(f"Start loading texts from {text_files_path}")
    docs = load_text_files(path=text_files_path)
    
    qa_test_data_path = qes_file_path
    qa_df = pd.read_csv(qa_test_data_path)
    # sample 100 rows from the dataframe
    qa_df = qa_df.sample(100, random_state=221)
    print(f"End loading texts. Number of documents for retrieval: {len(docs)}. Number of QA pairs: {len(qa_df)}")
    
    # Step 4: Split the documents into smaller chunks
    # Wrap text strings in Document objects
    documents = []
    for text in tqdm(docs, desc="wrapping text in Document objects"):
        documents.append(Document(page_content=text))
    del docs

    if splitter_type == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == "character":
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == "token":
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == "semantic":
        text_splitter = SemanticChunker(
            OpenAIEmbeddings(), 
            breakpoint_threshold_type="percentile", 
            number_of_chunks=chunk_size)
    else:
        print("Invalid splitter type. Please choose between recursive, character, token, or semantic.")
    
    splits = text_splitter.split_documents(documents)
    del documents
    print(f"End Spliting texts -- Number of splits: {len(splits)}")
    # Step 5: Create Chroma vectorstore with embeddings from Sentence Transformers
    # print(f"Start Embedding texts")
    embeddings = []
    for doc in tqdm(splits, desc="Embedding texts"):
        embeddings.append(embedding_model.encode(doc.page_content))
    embedding_wrapper = SentenceTransformerEmbeddings(embedding_model)
    print(f"End Embedding texts")
    # Free GPU cache after generating embeddings
    torch.cuda.empty_cache()

    # Step 6: Create the RAG prompting pipeline
    prompt_template = PromptTemplate(
        input_variables=['context', 'question'],
        template=PROMPT_TEMPLATE
    )

    # Update the HumanMessagePromptTemplate with the new PromptTemplate
    human_message_template = HumanMessagePromptTemplate(prompt=prompt_template)

    # Update the ChatPromptTemplate with the modified message
    chat_prompt_template = ChatPromptTemplate(
        input_variables=['context', 'question'],
        messages=[human_message_template]
    )

    prompt = chat_prompt_template

    # # Step 7: Load the QA test data
    # qa_test_data_path = qes_file_path
    # qa_df = pd.read_csv(qa_test_data_path)
    
    # # sample 100 rows from the dataframe
    # qa_df = qa_df.sample(100, random_state=42)
    
    # Step 7: Generate answers for the questions
    print("Building the vectorstore...")
    if retriever_type == "CHROMA":
        print("Building the vectorstore Chroma...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_wrapper, collection_name="collectionChroma")
        chroma_retriever = vectorstore.as_retriever(search_kwargs={'k': top_k_search})
        retriever = chroma_retriever
    elif retriever_type == "FAISS":
        print("Building FAISS...")
        embeddings_np = np.array(embeddings).astype("float32")
        faiss_retriever = FAISSRetriever(embeddings=embeddings_np, documents=splits)
        retriever = faiss_retriever
    else:
        print("Invalid retriever type. Please choose between FAISS or CHROMA.")
    
    print("Retriever built successfully!")
    torch.cuda.empty_cache()
    del splits
    
    answer_generation(
        qa_df, output_file, retriever_type, retriever, embedding_model, 
        generation_pipe, prompt, k=top_k_search)
    
    print(f"QA evaluation completed! Results saved to {output_file}")
