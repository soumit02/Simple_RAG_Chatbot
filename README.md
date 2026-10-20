# 📚 Simple PDF RAG Chatbot (Hybrid Search + Conversational Memory)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange)
![Search](https://img.shields.io/badge/Search-Hybrid_(Vector_+_BM25)-purple)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow?logo=huggingface&logoColor=black)
![Groq](https://img.shields.io/badge/LLM-Groq_API-f55036)

A modular **Retrieval-Augmented Generation (RAG)** CLI chatbot built with **LangChain**. Drop **any text-based PDF document(s)** into the `data/` folder, and the system automatically cleans and chunks the text, stores vector embeddings locally using **ChromaDB**, and lets you have an interactive, memory-aware conversation with your documents in real time.

Answers are grounded in the document only, and every retrieved passage carries its **file name and page number**, so the assistant can point you to where the information came from.

---

## ✨ Key Features

* **📂 Universal PDF Ingestion:** Automatically scans and loads single or multiple PDF files (including sub-folders) from the `data/` directory using `DirectoryLoader` without changing a single line of code.
* **🧹 Smart Automated Cleaning:** Handles the usual PDF noise using Regular Expressions (`re`):
  * Automatically detects and removes **repeated headers and footers** (per file, for PDFs with 5+ pages)
  * Removes `Page X of Y` lines and standalone page numbers
  * Repairs **words broken by hyphens** at line ends (`impor-\ntant` → `important`) and fixes **ligature characters** (`ﬁ` → `fi`)
  * Joins mid-sentence line breaks while keeping bullets, headings and paragraphs intact
  * Normalizes extra spaces and blank lines
* **✂️ Cross-Page Chunking with Page Tracking:** Pages of each file are merged *before* splitting with `RecursiveCharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=150`), so sentences and sections that continue across a page break stay together. The correct **page number** of every chunk is still preserved in its metadata. Tiny meaningless chunks are dropped.
* **🔎 Hybrid Retrieval (Vector + BM25):** Combines **semantic search** (meaning, via embeddings) with **BM25 keyword search** (exact words, names, labels) and merges both rankings using **Reciprocal Rank Fusion (RRF)**. This catches facts that pure semantic search can miss, such as a name or a label like "Submitted To".
* **🔁 History-Aware Question Rewriting:** Follow-up questions such as *"What was the date and topic of the second one?"* are automatically rewritten into standalone questions (*"…of Soumit's second workshop?"*) **before** retrieval, so the correct passages are found.
* **📌 Source-Aware Answers:** Retrieved chunks are passed to the LLM together with `[Source: file, Page: N]`, enabling answers that cite the page.
* **🆓 100% Free Local Embeddings:** Converts text chunks into 384-dimensional vectors using the open-source `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face.
* **💾 Persistent Vector Database:** Stores embeddings locally on disk using **ChromaDB** (`chroma_store/`), so documents are indexed once and reused on every run.
* **🧠 Conversational Memory:** Remembers the last 5 turns (10 messages) of the conversation using `MessagesPlaceholder`, allowing natural follow-up questions.
* **⚡ Real-Time Streaming:** Streams LLM responses word-by-word in the terminal using OpenAI-compatible API endpoints (such as **Groq**).
* **🐞 Debug Mode:** Set `RAG_DEBUG=1` to print the rewritten question and the retrieved chunks (with page numbers) for every query. Very helpful for understanding and tuning the retrieval.

---

## 🏗️ System Architecture (How It Works)

```text
[PDF Files in data/]
       │
       ▼
1. Cleaning (headers/footers, hyphens, page numbers) + merge pages of each file
       │
       ▼
2. Chunking (800 chars, 150 overlap) with page-number tracking
       │
       ▼
3. HuggingFace Embeddings (all-MiniLM-L6-v2) ──► Persistent ChromaDB (chroma_store/)
                                                          │
[User Question] ──► 4. Question Rewriting (uses chat history)
                                                          │
                                                          ▼
                    5. Hybrid Retriever (retriever.py)
                       ├─ Vector search (semantic)  ──┐
                       └─ BM25 search (keywords)    ──┴─► Reciprocal Rank Fusion ─► Top-6 chunks
                                                          │
                                                          ▼
                    6. LCEL RAG Chain (rag_chain.py): context with sources + chat history
                                                          │
                                                          ▼
                    7. LLM via ChatOpenAI / Groq API ──► [Streamed Answer in CLI (main.py)]
```

---

## 📁 Project Structure

```text
UNIVERSAL_RAG_PROJECT/
│
├── data/                        # Place your PDF files here
│   └── sample_document.pdf
│
├── src/                         # Core application source code
│   ├── ingest_pdf.py            # Step 1: Loads, cleans, merges, chunks & embeds PDFs into ChromaDB
│   ├── retriever.py             # Step 2: Hybrid retriever (vector search + BM25 + RRF)
│   ├── rag_chain.py             # Step 3: LLM, question rewriting, prompt template & LCEL chain with memory
│   └── main.py                  # Step 4: Interactive CLI chatbot entry point with streaming
│
├── chroma_store/                # Auto-generated local vector database directory (ignored in Git)
├── .env                         # Environment variables (API keys, Base URL, Model Name)
├── .gitignore                   # Files and directories ignored by Git
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🛠️ Tech Stack

* **Framework:** [LangChain](https://www.langchain.com/) (`langchain`, `langchain-core`, `langchain-community`)
* **Document Loader & Splitter:** `PyPDFLoader`, `DirectoryLoader`, `RecursiveCharacterTextSplitter`
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (`langchain-huggingface`)
* **Vector Store:** `ChromaDB` (`langchain-chroma`)
* **Keyword Search:** `BM25Retriever` (`langchain-community`) powered by `rank_bm25`
* **Rank Fusion:** Reciprocal Rank Fusion (implemented in `retriever.py`)
* **LLM Client:** `ChatOpenAI` (`langchain-openai`) configured with **Groq API** (`llama-3.3-70b-versatile`)
* **Environment Management:** `python-dotenv`

---

## 🚀 Getting Started (Installation & Setup)

Follow these step-by-step instructions to run the project on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/soumit02/Simple_RAG_Chatbot.git
cd Simple_RAG_Chatbot
```

### 2. Create and Activate a Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```


### 4. Configure Environment Variables (`.env`)
Create a `.env` file in the **root directory** of the project and add your API credentials:

```env
GROQ_API_KEY="your_groq_api_key_here"
BASE_URL="https://api.groq.com/openai/v1"
MODEL_NAME="your model"

# Optional: set to 1 to print the rewritten question and retrieved chunks
RAG_DEBUG=0
```
*(Note: Since this project uses `ChatOpenAI` with a custom `BASE_URL`, you can easily swap Groq with OpenAI, OpenRouter, or Together AI just by changing the `.env` values!)*





## 💻 Usage Guide

### Step 1: Add Your PDF Documents
Place any PDF file(s) you want to chat with inside the `data/` directory at the root of the project.

### Step 2: Ingest PDFs into Vector Database
Navigate to the `src/` directory and run the ingestion script **once** (or whenever you add/update PDFs in the `data/` folder):

```bash
cd src
python ingest_pdf.py
```
Each run rebuilds the database from scratch, so deleted PDFs never leave stale data behind.

**Example Output:**
```text
1. Scanning '../data' for PDF files...
   Loaded 9 pages from PDF file(s).
2. Cleaning text and merging pages of each file...
3. Splitting documents into chunks (size=800, overlap=150)...
   Produced 13 chunks.
4. Initializing Free Open-Source Embedding Model...
5. Creating and saving Vector Store in '../chroma_store'...
Done! Successfully stored 13 vectors.
```
If a PDF contains no extractable text (for example a scanned document), you will see a warning like `no text found in '<file>' (scanned PDF? needs OCR)`.

### Step 3: Start the Chatbot
Run the main CLI script from the `src/` directory to start chatting with your documents:

```bash
python main.py
```

### 💬 Example CLI Interaction
```text
=======================================================
        Universal PDF RAG Chatbot (With Memory)
   Ask questions from the PDFs in your 'data' folder.
   Type 'clear' to reset memory, or 'quit' to exit.
=======================================================

You: How many workshops did Soumit conduct?

Assistant: Soumit conducted **two workshops** during his internship – “Fundamentals of Embedded Programming with Arduino UNO” (Page 5) and “IoT Application Building Using ESP01S” (Page 6).
-------------------------------------------------------

You: What was the date and topic of the second one?

Assistant: The second workshop was titled **“IoT Application Building Using ESP01S.”** It took place on **September 17, 2025** (see Internship_Soumit.pdf, Page 6).
-------------------------------------------------------

You: quit
Goodbye!
```
Notice how the follow-up *"the second one"* works: the question is rewritten using the chat history before searching the document.

### 🐞 Debug Mode (Optional)
With `RAG_DEBUG=1` in your `.env`, each question also prints:

```text
[DEBUG] Rewritten question: What was the date and topic of Soumit’s second workshop?
[DEBUG] page=6 | 'Workshop 2: "IoT Application Building Using ESP01S"\n• Position: Trainer...'
...
```
Use it to check whether the right passages were retrieved. If the answer is wrong, this tells you whether the problem is in retrieval or in the LLM.

---

## ⚙️ Configuration & Tuning

| Setting | File | Default | Notes |
|---|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `ingest_pdf.py` | `800` / `150` | Keep chunks within the embedding model's input limit (~256 tokens for MiniLM) |
| `MIN_CHUNK_CHARS` | `ingest_pdf.py` | `40` | Chunks shorter than this are discarded |
| `EMBED_MODEL` | `ingest_pdf.py` **and** `retriever.py` | `all-MiniLM-L6-v2` | Must be identical in both files. Re-run ingestion after changing it |
| `FETCH_K` | `retriever.py` | `20` | Candidates fetched by each search method before fusion |
| `get_retriever(k=...)` | `rag_chain.py` | `6` | Final number of chunks sent to the LLM |
| `temperature` | `rag_chain.py` | `0.2` (answers), `0` (rewriting) | Lower = more factual |

**Tip:** For non-English or specialised documents, try a multilingual model such as `BAAI/bge-m3` or a stronger English model such as `BAAI/bge-small-en-v1.5`, then re-run `python ingest_pdf.py`.

---

## ⚠️ Known Limitations

This project works best on **text-based, single-column PDFs** (reports, articles, books, manuals). Be aware of the following:

* **Scanned PDFs (images):** No OCR is included, so scanned documents cannot be read. Run an OCR tool (e.g. OCRmyPDF) first.
* **Tables, formulas and multi-column layouts:** Text extraction by `PyPDFLoader` may mix up the reading order. Table-aware loaders are a planned improvement.
* **Whole-document questions:** Requests like *"Summarize the whole book"* or *"List everything mentioned"* need information from many places, but only the top-6 chunks are retrieved.
* **Charts and images:** Information inside images is not extracted.
* **Header/footer detection** only runs on PDFs with 5 or more pages, to avoid deleting real content from very short files.

---

## 🗺️ Roadmap

-  OCR support for scanned PDFs
-  Table-aware and layout-aware PDF parsing (e.g. PyMuPDF)
-  Reranker model for even more precise retrieval
-  Map-reduce summarization for chapter/book-level questions
-  Metadata filtering (ask questions about a specific file)

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.