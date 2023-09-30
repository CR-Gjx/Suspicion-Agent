from typing import Dict, List, Type

from langchain import chat_models, embeddings, llms
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLanguageModel

from setting import EmbeddingSettings, LLMSettings
from context import Context
from setting import Settings
from rich.console import Console
from agent import SuspicionAgent

def agi_init(
    agent_configs: List[dict],
    game_config:dict,
    console: Console,
    settings: Settings,
    user_idx: int = 0,
    webcontext=None,
) -> Context:
    ctx = Context(console, settings, webcontext)
    ctx.print("Creating all agents one by one...", style="yellow")
    for idx, agent_config in enumerate(agent_configs):
        agent_name = agent_config["name"]
        with ctx.console.status(f"[yellow]Creating agent {agent_name}..."):
            agent = SuspicionAgent(
                name=agent_config["name"],
                age=agent_config["age"],
                rule=game_config["game_rule"],
                game_name=game_config["name"],
                observation_rule=game_config["observation_rule"],
                status="N/A",  
                llm=load_llm_from_config(ctx.settings.model.llm),
            
                reflection_threshold=8,
            )
            for memory in agent_config["memories"]:
                agent.add_memory(memory)
        ctx.robot_agents.append(agent)
        ctx.agents.append(agent)
    
        ctx.print(f"Agent {agent_name} successfully created", style="green")

    ctx.print("Suspicion Agent started...")
   
    return ctx



# ------------------------- LLM/Chat models registry ------------------------- #
llm_type_to_cls_dict: Dict[str, Type[BaseLanguageModel]] = {
    "chatopenai": chat_models.ChatOpenAI,
    "openai": llms.OpenAI,
}

# ------------------------- Embedding models registry ------------------------ #
embedding_type_to_cls_dict: Dict[str, Type[Embeddings]] = {
    "openaiembeddings": embeddings.OpenAIEmbeddings
}


# ---------------------------------------------------------------------------- #
#                                LLM/Chat models                               #
# ---------------------------------------------------------------------------- #
def load_llm_from_config(config: LLMSettings) -> BaseLanguageModel:
    """Load LLM from Config."""
    config_dict = config.dict()
    config_type = config_dict.pop("type")

    if config_type not in llm_type_to_cls_dict:
        raise ValueError(f"Loading {config_type} LLM not supported")

    cls = llm_type_to_cls_dict[config_type]
    return cls(**config_dict)


def get_all_llms() -> List[str]:
    """Get all supported LLMs"""
    return list(llm_type_to_cls_dict.keys())


# ---------------------------------------------------------------------------- #
#                               Embeddings models                              #
# ---------------------------------------------------------------------------- #
def load_embedding_from_config(config: EmbeddingSettings) -> Embeddings:
    """Load Embedding from Config."""
    config_dict = config.dict()
    config_type = config_dict.pop("type")
    print(config)
    if config_type not in embedding_type_to_cls_dict:
        raise ValueError(f"Loading {config_type} Embedding not supported")

    cls = embedding_type_to_cls_dict[config_type]
    return cls(**config_dict)


def get_all_embeddings() -> List[str]:
    """Get all supported Embeddings"""
    return list(embedding_type_to_cls_dict.keys())
