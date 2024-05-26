from abc import ABC, abstractmethod
# EmbeddingFactory is an abstract class that defines the interface for creating Embeddings
class EmbeddingFactory(ABC):
    def __init__(self, model: str):
        self.model = model
        pass
    # Returns a EmbeddingFunction type from chromadb.api.types
    @abstractmethod
    def create_embedding_function(self):
        pass


from chromadb.api.types import (
    Documents,
    Embeddings,
    EmbeddingFunction
)
import ollama

class OllamaEmbeddingFactory(EmbeddingFactory):
    class EmbedOllama(EmbeddingFunction[Documents]):
        def __call__(self, documents: Documents) -> Embeddings:
            embeddings = []
            for document in documents:
                embedding = ollama.embeddings(model=self.model, prompt=document)
                embeddings.append(embedding.get("embedding"))
            return embeddings
    
    def create_embedding_function(self):
        return self.EmbedOllama()
    
    