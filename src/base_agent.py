from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv


class BaseAgent(ABC):
    def __init__(self) -> None:
        load_dotenv()
        os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
    
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
    