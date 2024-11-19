# DocWeaver-AI
## Introduction
- A advance RAG(Retrieval Augmented Generation) multi-LLM chatbot capable of querying from
  1. Document
  2. URLS
## How Advance RAG works
1. Gets query from user
2. Get url or document from user
3. Makes chunks , embedd and index them using FAISS
4. Generate 4 queries using LLMs from user query
5. Use system generated queries to rerank chunks relevent to original query and get top 3 documents
6. Answer original query using those reranked documents
## What you need to run code
1. install requirements using txt file in repository on python version 3.10
2. Get Gemini key from store it with name "Gemini_key" in keys.env
3. Download LLama llama3.2:1b from website "https://ollama.com"
