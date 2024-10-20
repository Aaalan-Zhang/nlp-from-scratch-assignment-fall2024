import os
import torch
import pandas as pd
from tqdm import tqdm

import faiss
import numpy as np

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
# from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document
from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    PromptTemplate
)
from sentence_transformers import SentenceTransformer

from utility.rag_utility import FAISSRetriever, SentenceTransformerEmbeddings, load_text_files, query_retriever, format_retreived_docs, load_qa_test_data, answer_generation, PROMPT_TEMPLATE

# ========================================
# Vars that can be set and read from another var file.
# ========================================
model_name = "meta-llama/Llama-3.1-8B-Instruct" 
# model_name = "unsloth/Llama-3.2-1B-Instruct"
# torchType = torch.bfloat16
torchType = torch.float16
padding_side = "left"
embeddingModelName = "sentence-transformers/all-MiniLM-L6-v2"
textFilesPath = "data/crawled/crawled_text_data"
qesFilePath = "data/annotated/generated_qa_pairs_3000_test20.csv"
topKSearch = 3
retrieverType = "FAISS" # "CHROMA" or "FAISS"
outputFile = "output/qa_rag_results.csv"



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

    # Step 1: Initialize the Hugging Face model as your LLM
    print("Initializing the Hugging Face model...")
    # model_name = "meta-llama/Llama-3.1-8B-Instruct" # TODO: model name to be replaced with the arg passed in
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torchType, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer, 
        torch_dtype=torchType
    )
    print("Model initialized successfully!")

    # Step 2: Load the Sentence Transformers model for embeddings
    # TODO: 
    # 1. model for embeddings can be a choice by the user
    # 2. truncate_dim to be replaced with the arg passed
    embedding_model = SentenceTransformer(embeddingModelName, truncate_dim=384)

    # Step 3: load the text files for building the index
    docs = load_text_files(path=textFilesPath) # TODO: path to be replaced with the arg passed in

    # Step 4: Split the documents into smaller chunks
    # Wrap text strings in Document objects
    documents = [Document(page_content=text) for text in docs]

    # TODO: 
    # 1. Type of text_splitter can be a choice by the user
    # 2. chunk_size and chunk_overlap to be replaced with the args passed in
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # text_splitter = RecursiveCharacterTextSplitter(
    #               separators=["\n\n", "\n", r"(?<=[.?!])\s+"],                                   
    #               keep_separator=True, is_separator_regex=True,
    #               chunk_size=1000, chunk_overlap=200)
    
    splits = text_splitter.split_documents(documents)

    # Step 5: Create Chroma vectorstore with embeddings from Sentence Transformers
    embeddings = [embedding_model.encode(doc.page_content) for doc in splits]
    embedding_wrapper = SentenceTransformerEmbeddings(embedding_model)

    print("Building the vectorstore...")
    # TODO: the method of building the vectorstore can be a choice by the user: dense retriever like FAISS, sparse retriever like BM25.
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_wrapper, collection_name="collectionChroma")

    num_documents = vectorstore._collection.count()
    print(f"Number of documents in the vectorstore: {num_documents}")

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

    # Step 7: Load the QA test data
    qa_test_data_path = qesFilePath # TODO: qa_test_data_path to be replaced with the arg passed in
    ref_doc_ids, questions, ref_answers = load_qa_test_data(qa_test_data_path)
    
    # Step 8: Generate answers for the questions
    if retrieverType == "CHROMA":
        chroma_retriever = vectorstore.as_retriever(search_kwargs={'k': topKSearch})
        retriever = chroma_retriever
    elif retrieverType == "FAISS":
        embeddings_np = np.array(embeddings).astype("float32")
        faiss_retriever = FAISSRetriever(embeddings=embeddings_np, documents=splits)
        retriever = faiss_retriever
    else:
        chroma_retriever = vectorstore.as_retriever(search_kwargs={'k': topKSearch})
        retriever = chroma_retriever
    
    generated_answers = answer_generation(questions, retriever, embedding_model, generation_pipe, prompt, k=topKSearch)
    
    # save the generated answers together with the questions and reference doc ids and answers
    qa_results = pd.DataFrame({
        "Ref Doc id": ref_doc_ids,
        "Question": questions,
        "Ref Answer": ref_answers,
        "Generated Answer": generated_answers,
    })
    
    # save the results to a csv file
    qa_results.to_csv(outputFile, index=False) # TODO: output_file to be replaced with the arg passed in
    print(f"QA evaluation completed! Results saved to {outputFile}")
