# functions.py

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

# Charger les variables d'environnement
load_dotenv(dotenv_path='env.env')

# Récupérer la clé API et le chemin du répertoire
api_key = os.getenv("OPENAI_API_KEY")
directory_path = os.getenv("DIRECTORY_PATH")

# Vérifier que la clé API est présente
if not api_key:
    raise ValueError("La clé API OpenAI n'est pas définie dans l'environnement.")
os.environ["OPENAI_API_KEY"] = api_key

# Vérifier que le chemin du répertoire est présent
if not directory_path:
    raise ValueError("Le chemin du répertoire n'est pas défini dans l'environnement.")

def load_files_from_directory(directory_path):
    data = []
    csv_data = []

    # Parcourir tous les fichiers dans le répertoire
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

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

def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    return all_splits

def create_vectorstore(all_splits):
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    return vectorstore

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

def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    return qa_chain

def ask_question(qa_chain, question):
    answer = qa_chain({"query": question})
    print(f"Question: {question}")
    print(f"Réponse: {answer}")
    return answer
