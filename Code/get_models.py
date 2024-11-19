from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain_openai import OpenAIEmbeddings



def get_models():
    load_dotenv("./keys.env")
    google_api_key = os.getenv("Gemini_key")
    google_ai = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    ollama_model = OllamaLLM(model="llama3.2:1b")
    # openai_model = OpenAI(model="gpt-3.5-turbo")
    return {
        "Gemini-1.5": google_ai,
        "Ollama": ollama_model,
        # "openai": openai_model
    }

def get_embeddings():
    load_dotenv("./keys.env")
    google_api_key = os.getenv("Gemini_key")
    google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=google_api_key)
    ollama_embeddings = OllamaEmbeddings(model="llama3.2:1b")
    # openai=OpenAIEmbeddings(model="gpt-3.5-turbo")

    return {
        "Gemini-1.5": google_embeddings,
        "Ollama": ollama_embeddings,
        # "OpenAI": openai
    }