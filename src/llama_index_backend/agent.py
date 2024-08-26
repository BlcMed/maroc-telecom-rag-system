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
        source = response.metadata
        return response, source


if __name__ == "__main__":
    import yaml
    CONFIG_FILE_PATH = './config/config.yml'
    with open(CONFIG_FILE_PATH, 'r') as file:
        config = yaml.safe_load(file)
    MODEL_NAME = config.get('MODEL_NAME')
    DATA_PATH = config.get('DATA_PATH')
    VECTOR_STORE_PATH_LLAMA_INDEX = config.get('VECTOR_STORE_PATH_LLAMA_INDEX')
    agent = LlamaIndexAgent(
        model = MODEL_NAME,
        data_path=DATA_PATH,
        vector_store_path=VECTOR_STORE_PATH_LLAMA_INDEX
    )
    question="Quels sont les critères pour la reconnaissance d'un élément de propriété, d'équipement et de matériel (PPE) comme actif selon les politiques comptables IFRS décrites ?"
    response = agent.question(question)
    print(type(response))
