"""
Step 3: Builds the Universal Conversational RAG Pipeline with ChatOpenAI and Memory.
Now with question rewriting, so follow-up questions like "the second one?" work.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from retriever import get_retriever

# Load .env file (since the code is in the src folder, we use '../.env')
load_dotenv('../.env')

token = os.getenv("GROQ_API_KEY")
endpoint = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")

# Set RAG_DEBUG=1 in your .env (or terminal) to see the rewritten question and retrieved pages
DEBUG = os.getenv("RAG_DEBUG", "0") == "1"

if not token:
    raise ValueError("API KEY environment variable not set.")

# LLM that writes the final answer (low temperature = more factual)
llm = ChatOpenAI(
    model_name=model_name,
    openai_api_key=token,
    openai_api_base=endpoint,
    temperature=0.2,
)

# LLM that only rewrites questions (temperature 0 = no creativity wanted here)
rewriter_llm = ChatOpenAI(
    model_name=model_name,
    openai_api_key=token,
    openai_api_base=endpoint,
    temperature=0,
)

SYSTEM_PROMPT = """You are a helpful and knowledgeable AI assistant.
Answer the user's question based ONLY on the provided context below and the conversation history.
Each context passage starts with its source (file name and page). When useful, mention the page number in your answer.

If the context does not contain enough information to answer the question, state clearly that the answer is not available in the provided document(s). Do not make up information.

Context:
{context}
"""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, rewrite the latest question "
     "so that it is fully standalone and understandable WITHOUT the chat history. "
     "Replace pronouns and vague references (e.g. 'it', 'he', 'the second one', 'that') "
     "with the actual names or topics from the history. "
     "Do NOT answer the question. If it is already standalone, return it unchanged. "
     "Output ONLY the rewritten question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

condense_chain = CONDENSE_PROMPT | rewriter_llm | StrOutputParser()


def format_docs(docs: list[Document]) -> str:
    """Formats retrieved chunks along with their file name and page number."""
    formatted_chunks = []
    for doc in docs:
        file_name = doc.metadata.get("file_name", "Document")
        page_num = doc.metadata.get("page", "?")
        formatted_chunks.append(f"[Source: {file_name}, Page: {page_num}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_chunks)


def build_rag_chain():
    """Creates a memory-aware RAG chain with question rewriting."""
    retriever = get_retriever(k=6)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def make_standalone(x: dict) -> str:
        """Rewrites a follow-up question into a standalone question."""
        if not x["chat_history"]:
            return x["question"]  # first question: nothing to rewrite
        try:
            rewritten = condense_chain.invoke(
                {"question": x["question"], "chat_history": x["chat_history"]}
            ).strip()
            return rewritten or x["question"]
        except Exception:
            return x["question"]  # if rewriting fails, fall back to the original

    def retrieve_context(x: dict) -> str:
        docs = retriever.invoke(x["standalone_question"])
        if DEBUG:
            print(f"\n[DEBUG] Rewritten question: {x['standalone_question']}")
            for d in docs:
                print(f"[DEBUG] page={d.metadata.get('page')} | {d.page_content[:80]!r}")
        return format_docs(docs)

    # Pipeline:
    # 1. Input dict: {"question": ..., "chat_history": ...}
    # 2. Add "standalone_question" (rewritten using the history)
    # 3. Add "context" (retrieved with the standalone question)
    # 4. prompt -> LLM -> plain string
    chain = (
        RunnablePassthrough.assign(standalone_question=RunnableLambda(make_standalone))
        | RunnablePassthrough.assign(context=RunnableLambda(retrieve_context))
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain