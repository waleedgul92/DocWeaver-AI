import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import langchain.vectorstores as vectorstore
import chromadb

def scrap_url_data(url):

    # Read the page source
    source = urlopen(url).read()

    # Make a soup object to parse the HTML
    soup = BeautifulSoup(source, 'html.parser')

    # Extract the plain text content from paragraphs
    paras = []
    for paragraph in soup.find_all('p'):
        paras.append(paragraph.text.strip())  # Strip removes leading/trailing whitespaces

    # Extract text from divs that contain headers and other content
    heads = []
    for head in soup.find_all('div', attrs={'class': 'mw-parser-output'}):
        heads.append(head.text.strip())  # Strip leading/trailing whitespaces

    # Combine paragraphs and headers, but don't interleave
    text = '\n\n'.join(paras)  # Join each paragraph with two new lines for separation

    # Drop footnote superscripts in brackets using regular expressions
    text = re.sub(r"\[.*?\]+", '', text)

    return text  # Returning separated paragraphs



def preprocess_document_data(document_path):
    pdf_loader = PyPDFLoader(document_path)
    pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)


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


    for text in range(len(texts)):
        texts[text] =remove_emojis(texts[text])


    return texts


def store_data(text,category):
    chroma_client=chromadb.PersistentClient('vectorstore')
    chroma_client.get_or_create_collection(name=category)
    for i, chunk in enumerate(text_chunks):
        metadata = {"source": source, "chunk_id": i}
        collection.add(text=chunk, metadata=metadata)

