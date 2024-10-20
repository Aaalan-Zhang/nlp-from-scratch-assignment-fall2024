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

# ========================================
# Main Script Execution
# ========================================
if __name__ == "__main__":

    # Step 0: Load environment variables
    # load_dotenv()

    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
    # os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # os.environ["LANGCHAIN_PROJECT"] = "rag-template"
    # os.environ["USER_AGENT"] = "LangChain/1.0 (+https://www.langchain.com)"

    # login(token=os.getenv('HUGGINGFACE_TOKEN'))

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
