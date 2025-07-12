# DocWeaver-AI

## Introduction

DocWeaver-AI is an advanced Retrieval Augmented Generation (RAG) multi-LLM chatbot. It is capable of querying from documents and URLs to answer user questions. The application leverages multiple large language models and a sophisticated RAG pipeline to provide accurate and contextually relevant answers.

## How the Advanced RAG Works

The advanced RAG process is as follows:

1.  The user provides a query and a document or URL.
2.  The application processes the input, creating chunks of the data, embedding them, and indexing them using FAISS.
3.  The system generates four new queries from the user's original query using an LLM.
4.  These generated queries are used to rerank the document chunks, and the top three most relevant chunks are selected.
5.  Finally, the system uses the reranked documents to answer the user's original query.

## Getting Started

### Prerequisites

  * Python 3.10
  * A Gemini API key
  * A Groq API key
  * Ollama with the `llama3.2:1b` model downloaded

### Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/waleedgul92/docweaver-ai.git
    cd docweaver-ai
    ```

2.  **Install dependencies:**
    Install the required Python packages using the `Requirements.txt` file.

    ```bash
    pip install -r Requirements.txt
    ```

3.  **Set up API keys:**

      * Create a file named `keys.env` in the root directory.
      * Add your Gemini API key to the `keys.env` file with the name "Gemini\_key".
      * Add your Groq API key to the `keys.env` file with the name "GROQ\_API\_KEY".

4.  **Download the Ollama model:**
    Download the Llama `llama3.2:1b` model from the [Ollama website](https://ollama.com).

### Running the Application

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run Code/main.py --server.enableXsrfProtection false
```
