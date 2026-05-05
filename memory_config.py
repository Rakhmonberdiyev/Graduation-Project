from mem0 import Memory
import os
from dotenv import load_dotenv
load_dotenv()

_api_key = os.getenv("OPENAI_API_KEY")
_neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
_qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
_neo4j_port = int(os.getenv("NEO4J_PORT", "7687"))

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": _api_key,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": _api_key,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": _qdrant_port,
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": f"bolt://localhost:{_neo4j_port}",
            "username": "neo4j",
            "password": _neo4j_password,
        }
    },
}

memory = Memory.from_config(config)
