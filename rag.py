from chromadb.api.types import (
    Documents,
    Embeddings,
    EmbeddingFunction
)
import chromadb
from crewai_tools import BaseTool

mode = "ollama"

# Create the client for the chromaDB
client = chromadb.PersistentClient(path="./chromadb")

import EmbeddingFactory

embedding_factory : EmbeddingFactory

if (mode == "ollama"):
    embedding_factory = EmbeddingFactory.OllamaEmbeddingFactory(model="mxbai-embed-large")


# Create the collection for the script
script_collection = client.get_or_create_collection(name="script_collection", embedding_function=embedding_factory.create_embedding_function())

# Open the transcript file 
with open("./transcripts/video-large.txt", "r") as file:
    #Read line by line
    i=script_collection.count()
    for line in file:
        # Add the line to the collection
        script_collection.add(
            documents=[line],
            metadatas=[{"source": "video-large.txt"}],
            ids=[f"id_{i}"]
        )
        i+=1

# Create the search tool for the script collection
class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search in French the document collection for the query texts and return the documents that are related with it."

    def _run(self, query_texts: List[str]) -> str:
        response = script_collection.query(
            query_texts=query_texts,
            n_results=1
        )
        print('response')
        print(response)
        return response.get("documents")
        
print("Collection filled")

# RAG With ChromaDB

# document_reader = Agent(
#     role="Script reader",
#     goal="Read the content of the document and extract the information requested.",
#     backstory="""
#       You're a script reader, you need to read the content of the document and extract the information requested.
#       The document contains the script of a movie, you need to search for the information requested in the script.
#       If you cannot find the information in the script, you need to say so.
#       You can use the script search tool to search the script for the answer as passing all the words associated with alcool in the query_texts (in French).
#     """,
#     verbose=True,
#     allow_delegation=False,
#     tools=[DocumentSearchTool(collection=script_collection)],
#     llm=model_analyst
# )