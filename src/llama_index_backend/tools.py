from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


def generate_tools(index, similarity_top_k, similarity_cutoff):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )
    #query_engine = index.as_query_engine()
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
    )
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine 
    )
    tools = [tool]
    return tools