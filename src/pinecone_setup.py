from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

pc.create_index(
    name="chatbot-agent",
    dimension=1536,  # OpenAI embedding dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

print("✅ Created Pinecone index with dimension=1536")
