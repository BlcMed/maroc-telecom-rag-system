from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


def load_index(vector_store_path):
    db = chromadb.PersistentClient(path = vector_store_path)
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )
    return index

def create_index(documents, vector_store_path, chunk_size, chunk_overlap):
    db = chromadb.PersistentClient(path=vector_store_path)
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    Settings.text_splitter = text_splitter

    # after we pass storage_context, chroma automatically saves data to disk
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        storage_context=storage_context,
        show_progress=True
    ) 
    return index
