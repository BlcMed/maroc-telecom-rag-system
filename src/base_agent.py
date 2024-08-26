from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv


class BaseAgent(ABC):
    def __init__(self, model, data_path, vector_store_path) -> None:
        load_dotenv()
        os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
        self.model=model
        self.data_path = data_path
        self.vector_store_path = vector_store_path
    
    @abstractmethod
    def question(self,prompt):
        """
        to ask any question concerning the documents
        Args: 
            prompt (str): the question
        Return:
            response (str): the response by the LLM
        """
        pass
    