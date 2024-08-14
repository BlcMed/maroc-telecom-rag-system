# agent.py

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

class LangChainAgent:
    def __init__(self, vectorstore):
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=vectorstore.as_retriever())

    def query(self, prompt):
        response = self.qa_chain({"query": prompt})
        return response
