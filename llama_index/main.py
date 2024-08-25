from .indexer import load_index
from .agent import Llama_agent
from dotenv import load_dotenv
import os

class LlamaIndexInterface:

    def __init__(self):
        load_dotenv()
        os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
        self.index = load_index()
        self.agent = Llama_agent(index=self.index)
        print(' llama interface initialized')

    def query(self, prompt):
        return self.agent.query(prompt)

if __name__ == "__main__":
    interface = LlamaIndexInterface()
    question="Quels sont les critères pour la reconnaissance d'un élément de propriété, d'équipement et de matériel (PPE) comme actif selon les politiques comptables IFRS décrites ?"
    response = interface.query(question)
    print(response)
