"""Claude Opus 4.7 lead-optimization agent.

Iteratively mutates a pair of designed α-helices bound to a target
structure, scoring each proposal against shape complementarity,
electrostatic complementarity, and buried surface area, and maintains
a Pareto frontier of non-dominated designs."""
from .config import AgentConfig, load_agent_config
from .designer import Designer, Prediction
from .pareto import Design, ParetoFrontier

__all__ = [
    "AgentConfig",
    "Design",
    "Designer",
    "ParetoFrontier",
    "Prediction",
    "load_agent_config",
]
