from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_openai.llms import OpenAI
# from langchain_openai import OpenAIEmbeddings
# from langchain_groq import ChatGroq




def get_models():
    load_dotenv("./keys.env")
    google_api_key = os.getenv("Gemini_key")
    google_ai = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    ollama_model = OllamaLLM(model="llama3.2:1b",base_url="http://localhost:11434" )
    # openai_model = OpenAI(model="gpt-3.5-turbo")
    # groq_llm=ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-1b-preview")
    return {
        "Gemini-1.5": google_ai,
        "Ollama": ollama_model
        # "openai": openai_model
        # "Groq_llm":g  roq_llm
    }

def get_embeddings():
    load_dotenv("./keys.env")
    google_api_key = os.getenv("Gemini_key")
    os.environ["GOOGLE_API_KEY"] = google_api_key
    google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    ollama_embeddings = OllamaEmbeddings(model="llama3.2:1b")
    # openai=OpenAIEmbeddings(model="gpt-3.5-turbo")

    return {
        "Gemini-1.5": google_embeddings,
        "Ollama": ollama_embeddings,
        # "OpenAI": openai
        # "Groq_llm": ollama_embeddings
    }