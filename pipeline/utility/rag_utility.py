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

# ========================================
# Custom Class for FAISS
# ========================================
class FAISSRetriever:
    """A FAISS retriever to handle vector search and document retrieval."""
    def __init__(self, embeddings, documents):
        self.documents = documents  # Store documents separately since FAISS does not handle metadata
        self.index = self._create_faiss_index(embeddings)

    def _create_faiss_index(self, embeddings):
        """
        Create a FAISS index from the embeddings.
        Args:
        embeddings (list): List of document embeddings.

        Returns:
        faiss.IndexFlatL2: FAISS index built from the embeddings.
        """
        d = len(embeddings[0])  # Dimensionality of the embeddings
        index = faiss.IndexFlatL2(d)  # Using L2 distance for similarity search
        index.add(embeddings)
        return index

    def retrieve(self, query_embedding, k=3):
        """
        Retrieve the top-k documents closest to the query embedding.
        Args:
        query_embedding (ndarray): The embedding of the query.
        k (int): The number of nearest neighbors to return.

        Returns:
        list: A list of the top-k documents.
        """
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]  # Return the documents corresponding to the nearest neighbors

# ========================================
# Custom Class for Sentence Transformer Embeddings
# ========================================
class SentenceTransformerEmbeddings:
    """A wrapper class for Sentence Transformers model to provide the same interface as OpenAIEmbeddings."""
    def __init__(self, model):
        self.model = model

    # Method to generate embeddings for a list of texts (documents)
    def embed_documents(self, texts):
        return [self.model.encode(text) for text in texts]

    # Method to generate embeddings for a single query (optional but useful)
    def embed_query(self, text):
        return self.model.encode(text)


# ========================================
# Helper Functions
# ========================================
def load_text_files(path):
    """
    Load text files from the given path.
    
    Args:
    path (str): The path to the directory containing the text files.
    
    Returns:
    list: A list of text documents.
    """
    
    docs = []

    # Check if the path is a directory
    if os.path.isdir(path):
        # Iterate over files in the directory
        for file_name in os.listdir(path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    docs.append(file.read())
    elif os.path.isfile(path) and path.endswith(".txt"):
        # If the path is a file, directly read it
        with open(path, 'r', encoding='utf-8') as file:
            docs.append(file.read())

    return docs

def query_retriever(query, retriever, embedding_model, k=topKSearch):
    # Retrieve context using the retriever
    if isinstance(retriever, Chroma):
        retrieved_docs = retriever.invoke(query)
    elif isinstance(retriever, FAISSRetriever):
        # Chroma's k is set when creating the retriever. Possibly can be changed dynamically with further research.
        query_embedding = embedding_model.encode(query).reshape(1, -1).astype("float32")
        retrieved_docs = faiss_retriever.retrieve(query_embedding, k=k)
    return retrieved_docs

def format_retreived_docs(docs):
    """
    Format the retrieved documents as the following:
    Context 1: <content of document n>
    Context 2: <content of document n-1> # in reverse order
    ...
    """
    # reverse the order of the documents
    docs = reversed(docs)
    return "\n\n".join(f"Context {i + 1}: {doc.page_content}" for i, doc in enumerate(docs))

def load_qa_test_data(path):
    """
    Load the QA test data from the given path.
    
    Args:
    path (str): The path to the QA test data file.
    
    Returns:
    pd.DataFrame: A DataFrame containing the test data.
    """
    
    qa_df = pd.read_csv(path)

    questions = qa_df["Question"].tolist()
    answers = qa_df["Answer"].tolist()
    doc_ids = qa_df["Doc_id"].tolist()

    # random sample 10 qa pairs for lightweight testing
    import random
    sample_size = 10
    random.seed(747)
    sample_indices = random.sample(range(len(questions)), sample_size)
    questions = [questions[i] for i in sample_indices]
    answers = [answers[i] for i in sample_indices]
    doc_ids = [doc_ids[i] for i in sample_indices]
    
    return doc_ids, questions, answers

def answer_generation(questions, retriever, embedding_model, generation_pipe, prompt, k=3):
    """
    Generate answers for the given questions using the retriever and the generation pipeline.
    
    Args:
    questions (list): A list of questions to answer.
    retriever (Chroma): A retriever object to retrieve documents.
    generation_pipe (pipeline): A pipeline object for text generation.
    prompt (ChatPromptTemplate): A ChatPromptTemplate object for generating prompts.
    
    Returns:
    list: A list of generated answers
    """
    generations = []
    print("Generating answers for the questions...")

    for question in tqdm(questions):
        # Retrieve documents based on the question
        retrieved_docs = query_retriever(question, retriever, embedding_model, k=k)
        # Format the documents
        context = format_retreived_docs(retrieved_docs)

        # Create the full prompt using the prompt template
        prompt_messages = prompt.format_messages(context=context, question=question)
        full_prompt = "\n".join(message.content for message in prompt_messages)
        
        # print(full_prompt)
        
        messages = [
        {"role": "user", "content": full_prompt},
        ]
        with torch.no_grad():
            llm_output = generation_pipe(messages, max_new_tokens=50)
        
        generations.append(llm_output[0]["generated_text"][1]['content'])
        
    return generations

# ========================================
# Constants and Configuration
# ========================================

PROMPT_TEMPLATE = """
You are an expert assistant answering factual questions about Pittsburgh or Carnegie Mellon University (CMU). 
Use the retrieved information to give a detailed and helpful answer. If the provided context does not contain the answer, leverage your pretraining knowledge to provide the correct answer. 
If you truly do not know, just say "I don't know."

Important Instructions:
- Answer concisely without repeating the question.
- Use the provided context if relevant; otherwise, rely on your pretraining knowledge.
- Do **not** use complete sentences. Provide only the word, name, date, or phrase that directly answers the question. For example, given the question "When was Carnegie Mellon University founded?", you should only answer "1900".

Examples:
Question: Who is Pittsburgh named after? 
Answer: William Pitt
Question: What famous machine learning venue had its first conference in Pittsburgh in 1980? 
Answer: ICML
Question: What musical artist is performing at PPG Arena on October 13? 
Answer: Billie Eilish

Context: \n\n {context} \n\n
Question: {question} \n\n
Answer:
"""