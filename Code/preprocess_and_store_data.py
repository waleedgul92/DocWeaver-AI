import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import langchain.vectorstores as vectorstore
import chromadb
import bs4

def scrap_url_data(url,model):

    # Read the page source
    loader=WebBaseLoader( web_paths=url,
                         bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header")
                         )))
    page=loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_text=text_splitter.split_text(page)




def preprocess_document_data(document_path,model):
    pdf_loader = PyPDFLoader(document_path)
    pages = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents=text_splitter.split_documents(pages)
    def remove_emojis(string):
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
        
        return emoji_pattern.sub(r'', string)


    return chunked_documents