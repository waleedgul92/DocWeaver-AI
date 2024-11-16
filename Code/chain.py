from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ollama,openai ,google_palm
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

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

def retrieval_chain(db ,document_chain,llm=None):
    retreival=db.as_retriever()
    if llm:
        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retreival
    )
        retrieval_chain=create_retrieval_chain(compression_retriever,document_chain)
        return retrieval_chain
    else:
        retrieval_chain=create_retrieval_chain(retreival,document_chain)
        return retrieval_chain