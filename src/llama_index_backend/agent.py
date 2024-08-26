from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from .data_manager import initialize_index
from .tools import generate_tools
from ..base_agent import BaseAgent


class LlamaIndexAgent(BaseAgent):

    def __init__(self,
                 model,
                 data_path,
                 vector_store_path,
                 chunk_size,
                 chunk_overlap,
                 similarity_top_k,
                 similarity_cutoff) -> None:

        super().__init__(
            model,
            data_path=data_path,
            vector_store_path=vector_store_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        #self.index = load_index(vector_store_path=self.vector_store_path)
        self.index = initialize_index(data_path=data_path, vector_store_path=vector_store_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        tools = generate_tools(index=self.index, similarity_top_k=similarity_top_k, similarity_cutoff=similarity_cutoff)
        llm = OpenAI(model= model)
        self.agent = OpenAIAgent.from_tools(
            tools, llm=llm, verbose=True
        )
        print('llama interface initialized')

    def question(self, prompt):
        response = self.agent.chat(prompt)
        source = response.metadata
        print(f"Source: {source}")
        return response, source


if __name__ == "__main__":
    import yaml
    CONFIG_FILE_PATH = './config/config.yml'
    with open(CONFIG_FILE_PATH, 'r') as file:
        config = yaml.safe_load(file)
    MODEL_NAME = config.get('MODEL_NAME')
    DATA_PATH = config.get('DATA_PATH')
    VECTOR_STORE_PATH_LLAMA_INDEX = config.get('VECTOR_STORE_PATH_LLAMA_INDEX')
    CHUNK_SIZE = config.get('CHUNK_SIZE')
    CHUNK_OVERLAP = config.get('CHUNK_OVERLAP')
    agent = LlamaIndexAgent(
        model = MODEL_NAME,
        data_path=DATA_PATH,
        vector_store_path=VECTOR_STORE_PATH_LLAMA_INDEX,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    #question="Quels sont les critères pour la reconnaissance d'un élément de propriété, d'équipement et de matériel (PPE) comme actif selon les politiques comptables IFRS décrites ?"
    question="Quelles sont les perspectives commerciales pour le secteur des télécommunications en Afrique?"
    response = agent.question(question)
    #print(response.count)
    print(f"response count: {dir(response.count)}")
    print(f"response.index: {dir(response.index)}")

    print(dir(response))
