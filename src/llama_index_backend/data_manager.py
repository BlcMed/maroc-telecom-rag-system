from llama_index.core import SimpleDirectoryReader
from .indexer import create_index, load_index
from pathlib import Path


def load_documents(data_path):
    documents = SimpleDirectoryReader(data_path).load_data()
    return documents

def initialize_index(data_path, vector_store_path, chunk_size, chunk_overlap):

    folder_path = Path(vector_store_path)
    if folder_path.is_dir():
        print(f"Loading index from {vector_store_path}")
        index = load_index(vector_store_path=vector_store_path)
    else:
        print(f"Creating index at {vector_store_path}")
        documents = load_documents(data_path=data_path)
        index = create_index(
            documents=documents,
            vector_store_path=vector_store_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
    return index
