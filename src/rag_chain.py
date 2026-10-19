"""
Step 3: Builds the Universal Conversational RAG Pipeline with ChatOpenAI and Memory.
"""
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from retriever import get_retriever

# Load .env file (since the code is in the src folder, we use '../.env')
load_dotenv('../.env')

token = os.getenv("GROQ_API_KEY")
endpoint = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")

if not token:
    raise ValueError("API KEY environment variable not set.")

# LLM Model setup
llm = ChatOpenAI(
    model_name=model_name,
    openai_api_key=token,
    openai_api_base=endpoint,
    temperature=0.5,
)

SYSTEM_PROMPT = """You are a helpful and knowledgeable AI assistant. 
Answer the user's question based ONLY on the provided context below and the conversation history.

If the context does not contain enough information to answer the question, state clearly that the answer is not available in the provided document(s). Do not make up information.

Context:
{context}
"""


def format_docs(docs: list[Document]) -> str:
    """Formats retrieved chunks along with their file name and page number."""
    formatted_chunks = []
    for doc in docs:
        file_name = doc.metadata.get("file_name", "Document")
        # PyPDFLoader uses 0-based page indexing, so we add 1 for human readability
        page_num = doc.metadata.get("page", 0) + 1
        formatted_chunks.append(f"\n{doc.page_content}")
        
    return "\n\n---\n\n".join(formatted_chunks)


def build_rag_chain():
    """Creates a clean, memory-aware RAG chain."""
    retriever = get_retriever(k=5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Easy LCEL Pipeline:
    # 1. Takes input dict {"question": ..., "chat_history": ...}
    # 2. Extracts "question", passes it to retriever, and formats the chunks as "context"
    # 3. Passes "context", "chat_history", and "question" into the prompt -> LLM -> String

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain