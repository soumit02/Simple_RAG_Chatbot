"""
Step 2: Connects to the saved ChromaDB vector store and returns a retriever.
"""
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

CHROMA_DIR = "../chroma_store"
COLLECTION_NAME = "universal_rag_collection"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_retriever(k: int = 5):
    """Initializes the embedding model and returns the Chroma retriever."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_DIR}'. Please run 'python ingest_pdf.py' first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore.as_retriever(search_kwargs={"k": k})