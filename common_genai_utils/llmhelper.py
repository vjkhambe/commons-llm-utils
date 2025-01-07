
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login, HfApi
from dotenv import load_dotenv
import os
import tempfile
import chromadb
import nest_asyncio
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langsmith import traceable
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma


from langchain_ollama import OllamaLLM, ChatOllama

hf_repo_id_llama70b = "meta-llama/Meta-Llama-3.1-70B-Instruct"

MODEL_NAME = "llama3.2:3b"

def get_model_id():
        
    # Load and run the model:
    
    hf_repo_id = "mistralai/Ministral-8B-Instruct-2410"

    hf_repo_id="meta-llama/Meta-Llama-3-8B-Instruct" 
    
    return hf_repo_id
    
def get_llm(model_name=""): 
    
    #llm = get_llm_from_huggingface()
    #if model_name == "gemini": 
    #return get_google_gemini_llm()
    return get_llm_from_ollama()

def get_llm_from_huggingface():
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    #api = HfApi()
    #api.set_access_token(hf_api_key, add_to_git_credential=True)
    login(token=hf_api_key, add_to_git_credential=True) # login to HuggingFace Hub   
    
    hf_repo_id = get_model_id()

    llm_endpoint = HuggingFaceEndpoint(repo_id=hf_repo_id)    
    llm =ChatHuggingFace(llm=llm_endpoint)
    return llm

def get_llm_from_ollama():
    llm = OllamaLLM(model=MODEL_NAME)
    #llm = ChatOllama(model=MODEL_NAME)
    return llm

def get_chatmodel_from_ollama():
    #llm = OllamaLLM(model=MODEL_NAME)
    llm = ChatOllama(model=MODEL_NAME)
    return llm


PERSIST_DIR = "./chroma_vectordb"
embeddings = OllamaEmbeddings(model=MODEL_NAME)




def retrieve_docs(weburl: str, is_sitemap_url=False): 
    

    nest_asyncio.apply()
    
    loader = None 
    if is_sitemap_url: 
        loader = SitemapLoader(web_path=weburl)
    else:
        loader = WebBaseLoader(web_path=weburl)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    print("No of documents retrieved : ", len(docs) )
    doc_splits = text_splitter.split_documents(docs)
    print("No of documents retrieved after doc_splits : ", len(doc_splits) )
    # Save to disk
    
    vectorstore = Chroma.from_documents(
                     documents=doc_splits,                 # Data
                     embedding=embeddings,    # Embedding model
                     persist_directory=PERSIST_DIR # Directory to save data
                     )
    vectorstore.persist()
    print("No of Documents stored  : " , len(doc_splits))
    return get_retriever()

def get_retriever(): 
    # Load from disk
    embeddings_func = create_langchain_embedding(embeddings)
    vectorstore_disk = Chroma(
                            persist_directory=PERSIST_DIR,       # Directory of db
                            embedding_function=embeddings_func   # Embedding model
                       )
    # Get the Retriever interface for the store to use later.
    # When an unstructured query is given to a retriever it will return documents.
    # Read more about retrievers in the following link.
    # https://python.langchain.com/docs/modules/data_connection/retrievers/
    #
    # Since only 1 document is stored in the Chroma vector store, search_kwargs `k`
    # is set to 1 to decrease the `k` value of chroma's similarity search from 4 to
    # 1. If you don't pass this value, you will get a warning.
    retriever = vectorstore_disk.as_retriever(lambda_mult=0)
    
    # Check if the retriever is working by trying to fetch the relevant docs related
    # to the word 'MMLU' (Massive Multitask Language Understanding). If the length is greater than zero, 
    # it means that the retriever is functioning well.
    print(len(retriever.get_relevant_documents("MMLU")))
    return retriever

def old_get_vector_db_retriever(llm_model):
    persist_path = os.path.join("./temp/", "union.parquet")
    print(" get_vector_db_retriever for model ", llm_model)
    if llm_model.model == "models/gemini-1.5-pro":
        embd = GoogleGenerativeAIEmbeddings(model=llm_model.model)
    else:
        embd = OllamaEmbeddings(model=llm_model.model)
    

    # If vector store exists, then load it
    if os.path.exists(persist_path):
        print("loading existing vectorstore ")
        vectorstore = SKLearnVectorStore(
            embedding=embd,
            persist_path=persist_path,
            serializer="parquet"
        )
        return vectorstore.as_retriever(lambda_mult=0)
    print("index LangSmith documents and create new vector store : started")
    # Otherwise, index LangSmith documents and create new vector store
    ls_docs_sitemap_loader = SitemapLoader(web_path="https://docs.smith.langchain.com/sitemap.xml")
    ls_docs = ls_docs_sitemap_loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(ls_docs)

    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embd,
        persist_path=persist_path,
        serializer="parquet"
    )
    vectorstore.persist()
    print("index LangSmith documents and create new vector store : completed ")
    return vectorstore.as_retriever(lambda_mult=0)

def get_retriever_old(llm_model): 
    embed_fun = create_langchain_embedding(embeddings)
    vectordb_client = chromadb.PersistentClient()
    vectordb = Chroma(embedding_function=embed_fun, client=vectordb_client)
    retriever = vectordb.as_retriever(search_kwargs={"filter":{"id":"1"}})
    return retriever
    