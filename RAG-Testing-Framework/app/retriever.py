# app/retriever.py

from embeddings import embed_query
import faiss


def retrieve_context(query):
    query_vector = embed_query(query)

    # Dummy retrieval logic (replace with FAISS search)
    retrieved_docs = ["Doc1 relevant text", "Doc2 relevant text"]

    return retrieved_docs