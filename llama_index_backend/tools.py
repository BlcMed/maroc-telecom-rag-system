from llama_index.core.tools import BaseTool, FunctionTool, QueryEngineTool


def generate_tools(index):
    query_engine = index.as_query_engine()
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine 
    )
    tools = [tool]
    return tools