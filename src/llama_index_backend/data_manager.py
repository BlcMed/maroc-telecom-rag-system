from llama_index.core import SimpleDirectoryReader
from .indexer import create_index, load_index
from pathlib import Path

folder_path = Path('path/to/your/folder')

def load_documents(data_path):
    documents = SimpleDirectoryReader(data_path).load_data()
    return documents

def initialize_index(data_path, vector_store_path, chunk_size, chunk_overlap):

    if folder_path.is_dir():
        index = load_index(vector_store_path=vector_store_path)
    else:
        documents = load_documents(data_path=data_path)
        index = create_index(
            documents=documents,
            vector_store_path=vector_store_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
    return index
