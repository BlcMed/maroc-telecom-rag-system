from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from .indexer import load_index
from .tools import generate_tools
from ..base_agent import BaseAgent

#MODEL_NAME='gpt-4'

class LlamaIndexAgent(BaseAgent):

    def __init__(self, model, data_path, vector_store_path):
        super().__init__(
            model,
            data_path=data_path,
            vector_store_path=vector_store_path
        )

        self.index = load_index(vector_store_path=self.vector_store_path)
        tools = generate_tools(index=self.index)
        llm = OpenAI(model= model)
        self.agent = OpenAIAgent.from_tools(
            tools, llm=llm, verbose=True
        )
        print('llama interface initialized')

    def question(self, prompt):
        response = self.agent.chat(prompt)
        return response


if __name__ == "__main__":
    agent = LlamaIndexAgent()
    question="Quels sont les critères pour la reconnaissance d'un élément de propriété, d'équipement et de matériel (PPE) comme actif selon les politiques comptables IFRS décrites ?"
    response = agent.question(question)
    print(response)
