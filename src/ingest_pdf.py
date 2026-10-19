"""
Step 1: Automatically loads all PDFs from the '../data' directory, 
cleans the text, splits it into chunks, and stores embeddings in ChromaDB.
Run from the 'src' folder: python ingest_pdf.py
"""
import os
import re
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Paths configured relative to the 'src' folder
DATA_DIR = "../data"
CHROMA_DIR = "../chroma_store"
COLLECTION_NAME = "universal_rag_collection"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def clean_text(raw_text: str) -> str:
    """Universal text cleaner for any PDF document."""
    # Remove standalone page numbers (lines that only contain digits)
    text = re.sub(r"^\s*\d+\s*$", "", raw_text, flags=re.MULTILINE)
    
    # Replace multiple spaces or tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)
    
    # Replace 3 or more consecutive newlines with double newlines (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created '{DATA_DIR}' folder. Please put your PDF files inside it and run again.")
        return

    print(f"1. Scanning '{DATA_DIR}' for PDF files...")
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    pages = loader.load()

    if not pages:
        print(f"No PDF files found in '{DATA_DIR}'. Please add at least one PDF.")
        return

    print(f"   Loaded {len(pages)} pages from PDF file(s).")

    print("2. Cleaning text across all pages...")
    cleaned_pages = []
    for page in pages:
        page.page_content = clean_text(page.page_content)
        # Keep only non-empty pages after cleaning
        if page.page_content:
            # Extract just the filename for cleaner metadata
            source_path = page.metadata.get("source", "Unknown")
            page.metadata["file_name"] = os.path.basename(source_path)
            cleaned_pages.append(page)

    print(f"3. Splitting documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(cleaned_pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"   Produced {len(chunks)} chunks.")

    print("4. Initializing Free Open-Source Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Clear old database if it exists so new PDFs don't mix with old deleted ones
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # [OPTIONAL] If you ever want to use a paid API Embedding (like OpenAI), 
    # comment out the line above and uncomment the lines below:
    # from langchain_openai import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    print(f"5. Creating and saving Vector Store in '{CHROMA_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done! Successfully stored {vectorstore._collection.count()} vectors.")


if __name__ == "__main__":
    main()