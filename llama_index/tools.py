from llama_index.core.tools import BaseTool, FunctionTool, QueryEngineTool


def generate_tools(query_engine):
    tool = QueryEngineTool.from_defaults(
        query_engine, name="...", description="..."
    )
    tools = [tool]
    return tools