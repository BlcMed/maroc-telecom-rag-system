from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent


MODEL_NAME='gpt-4'

def setup_agent(tools):
    llm = OpenAI(model=MODEL_NAME)
    agent = OpenAIAgent.from_tools(
        tools, llm=llm, verbose=True
    )
    return agent


if __name__=='__main__':
    agent = setup_agent(tools=[])
    response = agent.chat("what is Maroc Telecom")
    print(str(response))

class Llama_agent:
    def __init__(self, index):
        self.query_engine = index.as_query_engine()

    def query(self, prompt):
        response = self.query_engine.query(prompt)
        return response