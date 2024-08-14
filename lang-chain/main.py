
class LangChainInterface:

    def __init__(self):
        pass

    def query(self, prompt):
        return prompt

# usage example
if __name__ == "__main__":
    interface = LangChainInterface()
    question="Quels sont les critères pour la reconnaissance d'un élément de propriété, d'équipement et de matériel (PPE) comme actif selon les politiques comptables IFRS décrites ?"
    response = interface.query(question)
    print(response)
