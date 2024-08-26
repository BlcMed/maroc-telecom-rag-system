import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader

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
