from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ollama,openai ,google_palm
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

def prompt_template(context):
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
