from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import AzureChatOpenAI
import os

def setup_graph_chain(llm):
    """
    Sets up the Graph RAG chain using Neo4j.
    Returns None if credentials are missing or connection fails.
    """
    url = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([url, username, password]):
        print("Neo4j credentials not found. Graph RAG will be disabled.")
        return None

    try:
        graph = Neo4jGraph(
            url=url, 
            username=username, 
            password=password
        )
        
        # We need a separate LLM for Cypher generation if we want to be specific, 
        # but reusing the main LLM or creating a similar one is fine.
        # The notebook used a specific configuration, we try to match it.
        
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True # Required for newer langchain versions when using graph
        )
        return chain
    except Exception as e:
        print(f"Failed to initialize Neo4j Graph: {e}")
        return None
