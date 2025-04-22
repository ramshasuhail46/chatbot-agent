import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("chatbot-agent")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# PDF path
pdf_file_path = "temp.pdf"

# Extract text from PDF
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return [p.get_text().strip() for p in doc if p.get_text().strip()]

# 1. Extract and combine text from all PDF pages
pdf_pages = extract_text_from_pdf(pdf_file_path)
full_text = "\n".join(pdf_pages)

# 2. Split text into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
pdf_text_chunks = text_splitter.split_text(full_text)

print(f"🔹 Total chunks created: {len(pdf_text_chunks)}")

# 3. Generate embeddings for chunks
document_embeddings = embeddings.embed_documents(pdf_text_chunks)

# 4. Prepare Pinecone vectors
vectors = [{
    "id": f"doc-{i}",
    "values": embedding,
    "metadata": {"chunk_text": chunk}
} for i, (chunk, embedding) in enumerate(zip(pdf_text_chunks, document_embeddings))]

# 5. Upsert vectors to Pinecone
index.upsert(vectors=vectors, namespace="default")

print("✅ PDF text chunks embedded and uploaded to Pinecone.")
