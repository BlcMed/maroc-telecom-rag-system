import os
import logging
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


data_path = os.getenv("DIRECTORY_PATH")
MODEL_NAME = "gpt-3.5 turbo"

# Vérifier que le chemin du répertoire est présent
if not data_path:
    raise ValueError("Le chemin du répertoire n'est pas défini dans l'environnement.")

def load_files_from_directory(data_path):
    data = []
    csv_data = []

    # Parcourir tous les fichiers dans le répertoire
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)

        if filename.endswith('.pdf'):
            # Charger le fichier PDF
            loader = PyPDFLoader(file_path)
            data.extend(loader.load())
        elif filename.endswith('.csv'):
            # Charger le fichier CSV
            df = pd.read_csv(file_path)
            csv_data.append(df)

    print(f"{len(data)} documents PDF chargés")
    print(f"{len(csv_data)} fichiers CSV chargés")
    return data, csv_data

def create_vectorstore(data_path):
    data = load_files_from_directory(data_path=data_path)
    all_splits = _split_text(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    return vectorstore

def _split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    return all_splits

def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    return qa_chain

def similarity_search(vectorstore, question):
    docs = vectorstore.similarity_search(question)
    print(f"Nombre de documents similaires trouvés : {len(docs)}")
    return docs

def setup_multi_query_retriever(vectorstore):
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=ChatOpenAI(temperature=0)
    )
    return retriever_from_llm

def ask_question(qa_chain, question):
    answer = qa_chain({"query": question})
    print(f"Question: {question}")
    print(f"Réponse: {answer}")
    return answer
