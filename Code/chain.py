from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser

def prompt_template():
    prompt=ChatPromptTemplate.from_template("""
       Answer the following question based on only context provided.
       Think step by step before providing detailed answer.
        <context>
        {context}
                                            
        Question : {input}
       """)
    
    return  prompt


def stuff_documents_chain(llm , prompt):
    document_chain=create_stuff_documents_chain(llm , prompt)
    return document_chain

def retrieval_chain(db ,document_chain):
    retreival=db.as_retriever()
    retrieval_chain=create_retrieval_chain(retreival,document_chain)
    return retrieval_chain





def get_queries():
    
# RAG-Fusion
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    return prompt_rag_fusion
