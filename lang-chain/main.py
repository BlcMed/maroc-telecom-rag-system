from dotenv import load_dotenv
import os
from functions import create_qa_chain, create_vectorstore

class LangChainInterface:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("La clé API OpenAI n'est pas définie dans l'environnement.")
        os.environ["OPENAI_API_KEY"] = api_key
        data_path = os.getenv("DIRECTORY_PATH")
        if not data_path:
            raise ValueError("Le chemin du répertoire n'est pas défini dans l'environnement")
        print(' lang chain interface initialized')
        vectorstore = create_vectorstore(data_path=data_path)
        self.qa_chain = create_qa_chain(vectorstore=vectorstore)

    def query(self, prompt):
        qa_chain = self.qa_chain
        response = qa_chain({"query": prompt})
        print(f"Question: {prompt}")
        print(f"Réponse: {response}")
        return response

if __name__ == "__main__":
    interface = LangChainInterface()
    question="Quels sont les critères pour la reconnaissance d'un élément de propriété, d'équipement et de matériel (PPE) comme actif selon les politiques comptables IFRS décrites ?"
    response = interface.query(question)
    print(response)
