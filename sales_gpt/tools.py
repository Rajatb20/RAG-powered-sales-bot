from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import pickle
import os
from sales_gpt.graph import setup_graph_chain

def setup_knowledge_base(llm):
    """
    Sets up the vector knowledge base from docsearch.pkl.
    """
    pkl_path = 'docsearch.pkl'
    if not os.path.exists(pkl_path):
        print(f"Warning: {pkl_path} not found.")
        return None

    try:
        with open(pkl_path, 'rb') as f:
            docsearch = pickle.load(f)
        
        knowledge_base = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=docsearch.as_retriever()
        )
        return knowledge_base
    except Exception as e:
        print(f"Failed to load vector store: {e}")
        return None

def get_tools(llm):
    """
    Returns a list of tools available for the agent.
    """
    tools = []
    
    # 1. Vector Search Tool
    knowledge_base = setup_knowledge_base(llm)
    if knowledge_base:
        tools.append(
            Tool(
                name="SimilarProductSearch",
                func=knowledge_base.run,
                description="Utilizes the product database to answer questions about product features, descriptions, and standard info.",
            )
        )
    
    # 2. Graph RAG Tool (Optional)
    graph_chain = setup_graph_chain(llm)
    if graph_chain:
        tools.append(
            Tool(
                name="GraphProductSearch",
                func=graph_chain.run,
                description="Use this tool to find deep relationships between products, entities, or complex queries that the standard search misses.",
            )
        )
    
    return tools
