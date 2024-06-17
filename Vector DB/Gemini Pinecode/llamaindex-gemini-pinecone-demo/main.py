import os
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, download_loader
from llama_index.core import Settings

GOOGLE_API_KEY = "AIzaSyDinVYuxQRYYnIVbRZdzgOQsoN6I9cHWuM"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = Gemini()
embed_model = GeminiEmbedding(model_name="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512


# Query the index, send the context to Gemini, and wait for the response
gemini_response = query_engine.query("What does the author think about LlamaIndex?")
print(gemini_response)
