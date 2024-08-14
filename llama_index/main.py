from indexer import load_index
from agent import Llama_agent

class LlamaIndexInterface:

    def __init__(self):
        self.index = load_index()
        self.agent = Llama_agent(index=self.index)

    def query(self, prompt):
        return self.agent.query(prompt)

if __name__ == "__main__":
    interface = LlamaIndexInterface()
    question="Quels sont les critères pour la reconnaissance d'un élément de propriété, d'équipement et de matériel (PPE) comme actif selon les politiques comptables IFRS décrites ?"
    response = interface.query(question)
    print(response)
