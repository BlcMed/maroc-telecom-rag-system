from .functions import load_files_from_directory, split_text, create_vectorstore, similarity_search, setup_multi_query_retriever, create_qa_chain, ask_question
import os

def main():
    # Charger les fichiers
    data_path = os.getenv("DIRECTORY_PATH")

    # Créer la base de données vectorielle
    vectorstore = create_vectorstore(data_path=data_path)
    # Créer une chaîne de question-réponse
    qa_chain = create_qa_chain(vectorstore)

    # Exemple de recherche de similarité
    #question = "What is the work dress code for male employees?"
    #docs = similarity_search(vectorstore, question)

    # Configurer le récupérateur multi-requêtes
    #retriever = setup_multi_query_retriever(vectorstore)
    #unique_docs = retriever.get_relevant_documents(query=question)
    #print(f"Nombre de documents uniques récupérés : {len(unique_docs)}")

    return qa_chain

if __name__ == "__main__":
    qa_chain = main()

    # Poser une question après l'exécution de main()
    question = "Quels sont les critères pour la reconnaissance d'un élément de propriété, d'équipement et de matériel (PPE) comme actif selon les politiques comptables IFRS décrites ?"
    answer = ask_question(qa_chain, question)
    print(f"Réponse à la question '{question}': {answer}")
