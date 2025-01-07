from dotenv import load_dotenv
import os



from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(): 
    model_name = "models/gemini-1.5-pro"
    print("model name used - ", model_name)
    llm = ChatGoogleGenerativeAI(model=model_name)
    return llm

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_chroma import Chroma

def get_retriever(): 
    model_name = "models/embedding-001"
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    embed_fun = create_langchain_embedding(embeddings)
    vectordb_client = chromadb.PersistentClient()
    vectordb = Chroma(embedding_function=embed_fun, client=vectordb_client)
    retriever = vectordb.as_retriever(search_kwargs={"filter":{"id":"1"}})
    return retriever    