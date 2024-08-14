# main.py

from data_manager import DataManager
from indexer import load_index
from agent import LangChainAgent

class LangChainInterface:

    def __init__(self):
        self.data_manager = DataManager()
        self.documents, self.csv_data = self.data_manager.charger_documents(["path/to/your/files"])
        self.index = load_index(self.documents)
        self.agent = LangChainAgent(vectorstore=self.index)
        print('LangChain interface initialized')

    def query(self, prompt):
        return self.agent.query(prompt)

if __name__ == "__main__":
    interface = LangChainInterface()
    question="Quels sont les critères pour la reconnaissance d'un élément de propriété, d'équipement et de matériel (PPE) comme actif selon les politiques comptables IFRS décrites ?"
    response = interface.query(question)
    print(response)
