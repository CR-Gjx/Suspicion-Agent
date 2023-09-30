import os
import pickle
from langchain import FAISS
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.embeddings.base import Embeddings
import faiss
from  context import  Context
from model import load_embedding_from_config
import math
from langchain.docstore import InMemoryDocstore

# reference:
# https://python.langchain.com/en/latest/use_cases/agent_simulations/characters.html#create-a-generative-character
def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


# reference:
# https://python.langchain.com/en/latest/use_cases/agent_simulations/characters.html#create-a-generative-character
def create_new_memory_retriever(ctx: Context):
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = load_embedding_from_config(ctx.settings.model.embedding)
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )


class Retriever(TimeWeightedVectorStoreRetriever):
    embedding_model: Embeddings

    def faiss_path(self, path) -> str:
        return path + "/faiss"

    def mem_path(self, path) -> str:
        return path + "/memory.pickle"

    def try_load_memory(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False

        faiss_path = self.faiss_path(path)
        faiss: FAISS = self.vectorstore
        faiss.load_local(faiss_path, self.embedding_model)

        mem_path = self.mem_path(path)
        with open(mem_path, "rb") as mem_file:
            self.memory_stream = pickle.load(mem_file)

        return True

    def dump_memory(self, path: str) -> bool:
        faiss: FAISS = self.vectorstore
        faiss.save_local(self.faiss_path(path))
        with open(self.mem_path(path), "wb") as mem_file:
            pickle.dump(self.memory_stream, mem_file)
