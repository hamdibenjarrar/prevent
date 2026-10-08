"""Prevent: run any model's output years forward and keep the best futures."""
from .engine import Engine, Future, Node, Result, score_node
from .models import OllamaModel, OpenAICompatibleModel, ScriptedModel, from_spec

__version__ = "0.1.0"
__all__ = ["Engine", "Future", "Node", "Result", "score_node", "OllamaModel", "OpenAICompatibleModel", "ScriptedModel", "from_spec"]
