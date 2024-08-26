from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

VECTOR_STORE_PATH = './llama_index_backend/chroma_db'

def load_index():
    db = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )
    return index

