# chatbot.py
from langchain_community.llms import OpenAI  
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from retrieve_embeddings import retrieve_documents  # 👈 Import your RAG function
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

class Chatbot:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = OpenAI(openai_api_key=openai_api_key)
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory)

    def generate_response(self, user_input: str):
        # Retrieve relevant documents
        context_chunks = retrieve_documents(user_input)
        context = "\n\n".join(context_chunks)

        # Combine context + user question
        full_input = f"Context:\n{context}\n\nQuestion:\n{user_input}"

        # Generate the LLM response
        response = self.conversation.predict(input=full_input)
        return response


if __name__ == "__main__":
    chatbot = Chatbot()
    print("🤖 RAG Chatbot is ready! (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye! 👋")
            break
        response = chatbot.generate_response(user_input)
        print("Chatbot:", response)
