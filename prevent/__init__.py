"""Prevent: run any model's output years forward and keep the best futures."""
from .engine import Engine, Future, Node, Result, score_node
from .models import OllamaModel, OpenAICompatibleModel, ScriptedModel, from_spec
from .tree import TEMPLATE, TreeModel, check_tree, run_tree

__version__ = "0.2.0"
__all__ = [
    "TEMPLATE",
    "Engine",
    "Future",
    "Node",
    "OllamaModel",
    "OpenAICompatibleModel",
    "Result",
    "ScriptedModel",
    "TreeModel",
    "check_tree",
    "from_spec",
    "run_tree",
    "score_node",
]
