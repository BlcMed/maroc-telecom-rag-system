

class Llama_agent:
    def __init__(self, index):
        self.query_engine = index.as_query_engine()

    def query(self, prompt):
        response = self.query_engine.query(prompt)
        return response