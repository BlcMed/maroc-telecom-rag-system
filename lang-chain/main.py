from dotenv import load_dotenv
import os

class LangChainInterface:
    def __init__(self):
        load_dotenv()
        os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
        print(' lang chain interface initialized')

    def query(qa_chain, prompt):
        response = qa_chain({"query": prompt})
        print(f"Question: {prompt}")
        print(f"Réponse: {response}")
        return response

if __name__ == "__main__":
    interface = LangChainInterface()
    question="Quels sont les critères pour la reconnaissance d'un élément de propriété, d'équipement et de matériel (PPE) comme actif selon les politiques comptables IFRS décrites ?"
    response = interface.query(question)
    print(response)
