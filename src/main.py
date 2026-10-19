"""
Step 4: Universal CLI Chatbot entry point with Conversation Memory.
Run from the 'src' folder: python main.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import build_rag_chain


def main():
    print("=======================================================")
    print("        Universal PDF RAG Chatbot (With Memory)        ")
    print("   Ask questions from the PDFs in your 'data' folder.  ")
    print("   Type 'clear' to reset memory, or 'quit' to exit.    ")
    print("=======================================================\n")

    chain = build_rag_chain()
    chat_history = []

    while True:
        question = input("You: ").strip()
        if not question:
            continue
            
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
            
        if question.lower() == "clear":
            chat_history.clear()
            print("[Memory cleared! Starting a fresh conversation.]\n")
            continue

        print("\nAssistant: ", end="", flush=True)
        
        full_response = ""
        for chunk in chain.stream({
            "question": question,
            "chat_history": chat_history
        }):
            print(chunk, end="", flush=True)
            full_response += chunk
            
        print("\n" + "-" * 55 + "\n")

        # Save conversation history (keep last 10 messages / 5 turns)
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=full_response))
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]


if __name__ == "__main__":
    main()