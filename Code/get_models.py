from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings






def get_models():
    load_dotenv("./keys.env")
    google_api_key = os.getenv("Gemini_key")
    google_ai = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

    return {
        "Gemini-1.5": google_ai
    }

def get_embeddings():
    load_dotenv("./keys.env")
    google_api_key = os.getenv("Gemini_key")
    google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=google_api_key)

    return {
        "google_embeddings": google_embeddings
    }