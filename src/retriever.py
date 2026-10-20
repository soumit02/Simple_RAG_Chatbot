"""
Step 2: Connects to the saved ChromaDB vector store and returns a HYBRID retriever.
Hybrid = vector search (meaning) + BM25 (exact keywords), merged with Reciprocal Rank Fusion.
"""
import os
import re

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers import BM25Retriever

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

CHROMA_DIR = "../chroma_store"
COLLECTION_NAME = "universal_rag_collection"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # must be the same as in ingest.py

FETCH_K = 20   # how many candidates each method (vector / BM25) brings
RRF_C = 60     # standard constant for Reciprocal Rank Fusion

# Very common words are ignored by the keyword search
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was",
    "were", "be", "by", "with", "as", "at", "it", "this", "that", "what", "who",
    "whom", "which", "how", "did", "does", "do", "his", "her", "he", "she", "s",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop punctuation and stopwords."""
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def get_retriever(k: int = 6):
    """Returns a retriever object with an .invoke(query) method -> list of Documents."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_DIR}'. Please run 'python ingest.py' first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Load every stored chunk once, to build the keyword (BM25) index
    stored = vectorstore.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(stored["documents"], stored["metadatas"])
    ]
    if not all_docs:
        raise ValueError("The vector store is empty. Please run 'python ingest.py' first.")

    bm25 = BM25Retriever.from_documents(
        all_docs, k=FETCH_K, preprocess_func=tokenize
    )

    def hybrid_search(query: str) -> list[Document]:
        vector_docs = vectorstore.similarity_search(query, k=FETCH_K)
        keyword_docs = bm25.invoke(query)

        scores = {}   # chunk_index -> fused score
        by_key = {}   # chunk_index -> Document
        for ranked_list in (vector_docs, keyword_docs):
            for rank, doc in enumerate(ranked_list):
                key = doc.metadata.get("chunk_index", doc.page_content[:50])
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_C + rank + 1)
                by_key[key] = doc

        best_keys = sorted(scores, key=scores.get, reverse=True)[:k]
        return [by_key[key] for key in best_keys]

    return RunnableLambda(hybrid_search)