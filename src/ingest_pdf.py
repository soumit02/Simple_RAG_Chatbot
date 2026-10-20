"""
Step 1: Loads all PDFs from the '../data' directory, cleans the text,
merges pages of each file, splits it into chunks (keeping page numbers),
and stores embeddings in ChromaDB.
Run from the 'src' folder: python ingest.py
"""
import os
import re
import shutil
from bisect import bisect_right
from collections import Counter, defaultdict

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Paths configured relative to the 'src' folder
DATA_DIR = "../data"
CHROMA_DIR = "../chroma_store"
COLLECTION_NAME = "universal_rag_collection"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 reads at most ~256 tokens (~1000 chars), so 800 is a safe size.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 40          # drop tiny, meaningless chunks
MIN_PAGES_FOR_HEADER_DETECTION = 5

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def normalize_line(line: str) -> str:
    """Turns 'Report | Page 12' and 'Report | Page 13' into the same key."""
    return re.sub(r"\d+", "#", line.strip())


def find_repeated_lines(page_texts: list[str], min_ratio: float = 0.3) -> set[str]:
    """Finds lines (headers/footers) that repeat on many pages of the same file."""
    if len(page_texts) < MIN_PAGES_FOR_HEADER_DETECTION:
        return set()  # too few pages to judge safely

    counter = Counter()
    for text in page_texts:
        keys = {normalize_line(l) for l in text.split("\n") if len(l.strip()) > 3}
        counter.update(keys)

    limit = len(page_texts) * min_ratio
    return {key for key, count in counter.items() if count >= limit and len(key) < 100}


def remove_repeated_lines(text: str, repeated: set[str]) -> str:
    if not repeated:
        return text
    kept = [l for l in text.split("\n") if normalize_line(l) not in repeated]
    return "\n".join(kept)


def clean_text(raw_text: str) -> str:
    """Universal text cleaner for any text-based PDF."""
    text = raw_text

    # Fix ligature characters (e.g. 'ﬁnance' -> 'finance')
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)

    # Remove trailing spaces at the end of each line
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Join words broken by a hyphen at line end: 'impor-\ntant' -> 'important'
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)

    # Remove lines like 'Page 12' or 'Page 12 of 300'
    text = re.sub(r"(?im)^[ \t]*page[ \t]+\d+([ \t]+of[ \t]+\d+)?[ \t]*$", "", text)

    # Remove standalone page numbers (lines that contain only 1-4 digits)
    text = re.sub(r"^[ \t]*\d{1,4}[ \t]*$", "", text, flags=re.MULTILINE)

    # Join lines that are the middle of a sentence (next line starts lowercase).
    # Bullets, headings and new sentences are kept on their own lines.
    text = re.sub(r"(?<=[^\n.!?:])\n(?=[a-z])", " ", text)

    # Replace multiple spaces or tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace 3 or more consecutive newlines with a paragraph break
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
        show_progress=True,
    )
    pages = loader.load()

    if not pages:
        print(f"No PDF files found in '{DATA_DIR}'. Please add at least one PDF.")
        return

    print(f"   Loaded {len(pages)} pages from PDF file(s).")

    # Group pages by file, in page order
    pages_by_file = defaultdict(list)
    for page in pages:
        file_name = os.path.basename(page.metadata.get("source", "Unknown"))
        page_no = page.metadata.get("page", 0) + 1  # 1-based for humans
        pages_by_file[file_name].append((page_no, page.page_content))

    print("2. Cleaning text and merging pages of each file...")
    merged_docs = []
    page_starts_by_file = {}  # file_name -> (list of start offsets, list of page numbers)

    for file_name, page_list in pages_by_file.items():
        page_list.sort(key=lambda p: p[0])

        # Detect repeated headers/footers inside this file
        repeated = find_repeated_lines([text for _, text in page_list])

        full_text = ""
        starts, numbers = [], []
        for page_no, raw in page_list:
            cleaned = clean_text(remove_repeated_lines(raw, repeated))
            if not cleaned:
                continue  # skip empty pages (image-only pages etc.)
            if full_text:
                full_text += "\n\n"
            starts.append(len(full_text))
            numbers.append(page_no)
            full_text += cleaned

        if not full_text:
            print(f"   WARNING: no text found in '{file_name}' (scanned PDF? needs OCR).")
            continue

        merged_docs.append(Document(page_content=full_text, metadata={"file_name": file_name}))
        page_starts_by_file[file_name] = (starts, numbers)

    if not merged_docs:
        print("No readable text found in any PDF.")
        return

    print(f"3. Splitting documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,  # remembers where each chunk starts in the merged text
    )
    raw_chunks = splitter.split_documents(merged_docs)

    chunks = []
    for chunk in raw_chunks:
        if len(chunk.page_content.strip()) < MIN_CHUNK_CHARS:
            continue
        # Find which page this chunk starts on
        starts, numbers = page_starts_by_file[chunk.metadata["file_name"]]
        idx = bisect_right(starts, chunk.metadata["start_index"]) - 1
        chunk.metadata["page"] = numbers[max(idx, 0)]
        chunk.metadata["chunk_index"] = len(chunks)
        chunks.append(chunk)

    print(f"   Produced {len(chunks)} chunks.")

    print("4. Initializing Free Open-Source Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Clear old database so new PDFs don't mix with old deleted ones
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # [OPTIONAL] To use a paid API embedding (like OpenAI), replace the line above with:
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