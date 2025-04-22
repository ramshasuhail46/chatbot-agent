# retrieve_embeddings.py
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("chatbot-agent")
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


def retrieve_documents(query, top_k=5):
    print(f"\n🔎 Searching Pinecone for: {query}")

    # Generate query embedding
    query_vector = embeddings.embed_query(query)
    # Print the first 10 elements of the query vector for debugging
    print(f"🔹 Query vector: {query_vector[:10]}...")

    # Query Pinecone using keyword arguments (not positional arguments)
    results = index.query(
        vector=query_vector,  # Use the query vector
        top_k=top_k,  # Number of top results
        include_metadata=True,  # Include metadata in the response
        include_values=True,  # Include the vector values in the response
        namespace="default",  # Using the default namespace, replace if necessary
        # Apply filters if needed, for example {"genre": {"$eq": "action"}}
        filter=None,
        score_threshold=0.8
    )

    matches = results.get("matches", [])
    print(f"✅ Retrieved {len(matches)} matches from Pinecone.")

    if not matches:
        print("⚠️ Pinecone returned 0 matches.")
        return []

    # Show detailed results for each match
    for match in matches:
        score = match.get("score", 0)
        text = match.get("metadata", {}).get("chunk_text", "")
        print(f"🧠 Score: {score:.4f}\nText: {text[:300]}...\n{'-'*50}")

    # Return the chunk_text of the best matches
    return [match['metadata']['chunk_text'] for match in matches]


# # Optional: enable testing from this file
# if __name__ == "__main__":
#     retrieved = retrieve_documents("What is World Food Safety Day?")

#     print("\n📄 Retrieved Chunks:\n", retrieved, sep="\n\n")
