from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from .tools import generate_tools


MODEL_NAME='gpt-4'

def setup_agent(index):
    tools = generate_tools(index=index)
    llm = OpenAI(model=MODEL_NAME)
    agent = OpenAIAgent.from_tools(
        tools, llm=llm, verbose=True
    )
    return agent


if __name__=='__main__':
    agent = setup_agent(index=None)
    response = agent.chat("what is Maroc Telecom")
    print(str(response))