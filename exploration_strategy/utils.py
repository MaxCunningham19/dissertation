import json
import os
from .Greedy import Greedy
from .ExplorationStrategy import ExplorationStrategy
from .Boltzmann import Boltzmann
from .DecayBoltzmann import DecayBoltzmann
from .EpsilonGreedy import EpsilonGreedy
from .DecayEpsilonGreedy import DecayEpsilonGreedy
from .Pheromones import Pheromones
from .DecayPheromones import DecayPheromones
from .Counts import Counts
from .GrowPheromones import GrowPheromones


strategies = {
    "boltzmann": Boltzmann,
    "decay_boltzmann": DecayBoltzmann,
    "epsilon": EpsilonGreedy,
    "decay_epsilon": DecayEpsilonGreedy,
    "pheromones": Pheromones,
    "decay_pheromones": DecayPheromones,
    "greedy": Greedy,
    "counts": Counts,
    "grow_pheromones": GrowPheromones,
}


def create_exploration_strategy(strategy_name: str, **kwargs) -> ExplorationStrategy:
    """Creates an instance of an exploration strategy given its name."""

    if strategy_name not in strategies:
        raise ValueError(f"Unknown exploration strategy: {strategy_name}")

    if "state_map" in kwargs:
        kwargs["state_map"] = StateMapGenerator.get_function(kwargs["state_map"])

    return strategies[strategy_name](**kwargs)  # initialize with provided arguments


def create_exploration_strategy_from_file(path: str) -> ExplorationStrategy:
    """Creates an instance of an exploration strategy given its name."""

    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            info_dict = json.load(f)
        strategy_name = info_dict.pop("strategy_name")

        if "state_map" in info_dict:
            info_dict["state_map"] = StateMapGenerator.get_function(info_dict["state_map"])

        return strategies[strategy_name](**info_dict)  # initialize with provided arguments
    except Exception as e:
        return None


class StateMapGenerator:
    """This class converts a string input into the desired state mapping function"""

    @classmethod
    def get_function(cls, function_string: str):
        """Returns the function corresponding to the passed string if it exists"""
        function_string = function_string.lower()
        if hasattr(cls, function_string):
            func = getattr(cls, function_string)
            if callable(func):
                return func

        # Get a list of all callable methods excluding 'get_function'
        available_functions = [name for name in dir(cls) if callable(getattr(cls, name)) and not name.startswith("__") and name != "get_function"]
        available_functions = "\n  - " + ("\n  -".join(available_functions))
        raise ValueError(f"Unsuppored state_map value: {function_string}. Availaible functions: {available_functions}")

    @staticmethod
    def tuple_map(state):
        return tuple(state)

    @staticmethod
    def do_nothing(state):
        return state
