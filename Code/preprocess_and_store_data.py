from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS  
import faiss
from urllib.parse import urlparse, urljoin
from langchain.schema import Document
import bs4

def scrap_url_data(url):
    # Parse and validate URL
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = urljoin("https://", url)  # Default to https if no scheme provided

    # Read the page source
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(
            # class_=("post-title","post-content","post-header")
        ))
    )
    
    page = loader.load()

    # Check if `page` is a list and extract text
    if isinstance(page, list):
        page_text = " ".join([doc.page_content for doc in page if hasattr(doc, "page_content")])
    elif isinstance(page, str):
        page_text = page
    else:
        raise ValueError("Unexpected page format: expected a list or string.")
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_text = text_splitter.split_text(page_text)

    return chunked_text


def preprocess_document_data(document_path):
    pdf_loader = PyPDFLoader(document_path)
    pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    

    texts = text_splitter.split_text(context)

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" # emoticons
        u"\U0001F300-\U0001F5FF" # symbols & pictographs
        u"\U0001F680-\U0001F6FF" # transport & map symbols
        u"\U0001F1E0-\U0001F1FF" # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    for i in range(len(texts)):
        texts[i] = emoji_pattern.sub(r'', texts[i])
    return texts

def generate_embedding_and_store(text, embedding_method):
    # Wrap each string in `text` as a Document with a page_content attribu

    documents = [Document(page_content=chunk) for chunk in text]
    db = FAISS.from_documents(documents, embedding_method)
    return db

def query_from_db(query ,db):
    retreived_result=db.similarity_search(query)
    return  retreived_result[0].page_content


