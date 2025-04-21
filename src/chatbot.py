# chatbot.py
from langchain_community.llms import OpenAI  # Corrected import
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Chatbot:
    def __init__(self):
        # Correctly initialize the OpenAI model using OpenAI class
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Instantiate the OpenAI LLM correctly
        self.llm = OpenAI(openai_api_key=openai_api_key)
        
        # Set up the conversation chain
        self.conversation = ConversationChain(llm=self.llm)

    def generate_response(self, user_input: str):
        # Generate response based on the user input
        response = self.conversation.predict(input=user_input)
        return response

# # Instantiate the Chatbot class
# chatbot = Chatbot()

# def main():
#     print("Chatbot: Hello! How can I assist you today?")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             print("Chatbot: Goodbye!")
#             break
#         response = chatbot.generate_response(user_input)
#         print("Chatbot:", response)

# if __name__ == "__main__":
#     main()
