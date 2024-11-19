from langchain_core.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Template for the answer generation prompt
def prompt_template():
    prompt = ChatPromptTemplate.from_template("""
       Answer the following question based on only context provided.
       Think step by step before providing a detailed answer.
        <context>
        {context}
                                            
        Question: {question}
       """)
    return prompt

# Template for generating search queries in RAG
def get_queries_template():
    template = """You are a helpful assistant that generates multiple search queries based on a single input query.
    Generate multiple search queries related to: {question}
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    return prompt_rag_fusion

# Generate queries using LLM and prompt template
def generate_queries(llm, prompt_rag_fusion, question):
    query_chain = prompt_rag_fusion | llm | StrOutputParser() | (lambda x: x.split("\n"))
    return query_chain.invoke({"question": question})

# Reciprocal Rank Fusion function
def reciprocal_rank_fusion(results: list[list], k=3):
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

# Define retrieval chain with RAG fusion
def retrieval_chain_rag_fusion(generate_queries, retriever, question):
    # Step 1: Generate queries
    query_list = generate_queries  # This should be the list of queries

    # Step 2: Retrieve documents for each query using FAISS retriever
    results = [retriever.get_relevant_documents(query) for query in query_list]

    # Step 3: Apply reciprocal rank fusion
    fused_results = reciprocal_rank_fusion(results)

    return fused_results

# Final RAG-based answer generation chain
def final_rag(retrieval_chain_rag_fusion_var, prompt, llm, question):
    # Combine context from retrieved documents
    context = "\n".join([doc.page_content for doc, score in retrieval_chain_rag_fusion_var])

    # Generate the prompt with context and question
    formatted_prompt = prompt.invoke({"context": context, "question": question}).to_string()
    
    # Generate the answer using the LLM
    answer_text = llm(formatted_prompt)
    parsed_answer = StrOutputParser().parse(answer_text)
    
    return parsed_answer
