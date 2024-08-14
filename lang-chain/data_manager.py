# data_manager.py

import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader

class DataManager:
    def __init__(self):
        pass

    def charger_documents(self, fichiers):
        data = []
        csv_data = []
        for filename in fichiers:
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(filename)
                data.extend(loader.load())
            elif filename.endswith('.csv'):
                df = pd.read_csv(filename)
                csv_data.append(df)
        return data, csv_data
