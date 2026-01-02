"""
LaCAM2 - Multi-Agent Path Finding Solver
Main solving interface
"""

import random
from typing import Optional

from .instance import Instance, Solution
from .planner import Planner, Objective
from .utils import Deadline


def solve(ins: Instance, 
          additional_info: str = "",
          verbose: int = 0,
          deadline: Optional[Deadline] = None,
          mt: Optional[random.Random] = None,
          objective: Objective = Objective.OBJ_NONE,
          restart_rate: float = 0.001) -> Solution:
    """
    Main solving function for LaCAM2
    
    Args:
        ins: Problem instance
        additional_info: Additional information string
        verbose: Verbosity level
        deadline: Time deadline
        mt: Random number generator
        objective: Optimization objective
        restart_rate: Restart rate for random restarts
    
    Returns:
        Solution if found, empty list otherwise
    """
    # Create planner
    planner = Planner(ins, deadline, mt, verbose, objective, restart_rate)
    
    # Solve
    # Solve
    solution = planner.solve(additional_info)
    
    return solution


__all__ = [
    'solve',
    'Instance',
    'Solution',
    'Objective',
    'Deadline',
]
