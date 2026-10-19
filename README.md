# 📚 Universal PDF RAG Chatbot (with Conversational Memory)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow?logo=huggingface&logoColor=black)
![Groq](https://img.shields.io/badge/LLM-Groq_API-f55036)

A modular, production-ready **Retrieval-Augmented Generation (RAG)** CLI chatbot built with **LangChain**. This project allows users to drop **any PDF document(s)** into the `data/` folder, automatically clean and chunk the text, store vector embeddings locally using **ChromaDB**, and have an interactive, memory-aware conversation with the documents in real-time.

---

## ✨ Key Features

* **📂 Universal PDF Ingestion:** Automatically scans and loads single or multiple PDF files from the `data/` directory using `DirectoryLoader` without changing a single line of code.
* **🧹 Automated Data Cleaning:** Cleans raw PDF text by removing standalone page numbers, fixing broken lines, and stripping excessive whitespaces using Regular Expressions (`re`).
* **✂️ Smart Text Chunking:** Uses `RecursiveCharacterTextSplitter` (`chunk_size=600`, `chunk_overlap=100`) to preserve semantic context across paragraphs and sentences.
* **🆓 100% Free Local Embeddings:** Converts text chunks into 384-dimensional vectors using the open-source `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face.
* **💾 Persistent Vector Database:** Stores embeddings locally on disk using **ChromaDB** (`chroma_store/`), eliminating the need to re-index documents on every run.
* **🧠 Conversational Memory:** Remembers the last 5 turns (10 messages) of the conversation using `MessagesPlaceholder`, allowing natural follow-up questions.
* **⚡ Real-Time Streaming:** Streams LLM responses word-by-word in the terminal using OpenAI-compatible API endpoints (such as **Groq**).

---

## 🏗️ System Architecture (How It Works)

```text
[PDF Files in data/] 
       │
       ▼
1. Data Cleaning & Chunking (ingest_pdf.py)
       │
       ▼
2. HuggingFace Embeddings (all-MiniLM-L6-v2)
       │
       ▼
3. Persistent ChromaDB (chroma_store/) ◄───┐
                                           │ (Similarity Search: Top-K Chunks)
[User Question] ──► 4. Retriever (retriever.py)
       │
       ▼
5. LCEL RAG Chain + Chat History (rag_chain.py)
       │
       ▼
6. LLM via ChatOpenAI / Groq API ──► [Streamed Answer in CLI (main.py)]
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
│   ├── ingest_pdf.py            # Step 1: Loads, cleans, chunks & embeds PDFs into ChromaDB
│   ├── retriever.py             # Step 2: Connects to ChromaDB and retrieves top-k relevant chunks
│   ├── rag_chain.py             # Step 3: Configures LLM, prompt template, and LCEL chain with memory
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
```
*(Note: Since this project uses `ChatOpenAI` with a custom `BASE_URL`, you can easily swap Groq with OpenAI, OpenRouter, or Together AI just by changing the `.env` values!)*

---

## 💻 Usage Guide

### Step 1: Add Your PDF Documents
Place any PDF file(s) you want to chat with inside the `data/` directory at the root of the project.

### Step 2: Ingest PDFs into Vector Database
Navigate to the `src/` directory and run the ingestion script **once** (or whenever you add/update PDFs in the `data/` folder):

```bash
cd src
python ingest_pdf.py
```
**Expected Output:**
```text
1. Scanning '../data' for PDF files...
   Loaded 9 pages from PDF file(s).
2. Cleaning text across all pages...
3. Splitting documents into chunks (size=600, overlap=100)...
   Produced 19 chunks.
4. Initializing Free Open-Source Embedding Model...
5. Creating and saving Vector Store in '../chroma_store'...
Done! Successfully stored 19 vectors.
```

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

You:  Where did Soumit Dey complete his internship and who supervised him?

Assistant: Soumit Dey completed his three‑month internship at **eChithi**.  
He worked under the direct supervision of **Md. Rakib Khan**, who is the **Business Development Manager at eChithi**.
-------------------------------------------------------

You: How many workshops did he conduct there?

Assistant: According to the internship report, Soumit Dey organized **two workshops** during his time at eChithi.
-------------------------------------------------------


You: quit
Goodbye!
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.