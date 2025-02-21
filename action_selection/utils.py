from .ExplorationStrategy import ExplorationStrategy
from .Boltzmann import Boltzmann
from .DecayBoltzmann import DecayBoltzmann
from .EpsilonGreedy import EpsilonGreedy
from .DecayEpsilonGreedy import DecayEpsilonGreedy
from .Pheromones import Pheromones


def create_exploration_strategy(strategy_name: str, *args, **kwargs) -> ExplorationStrategy:
    """Creates an instance of an exploration strategy given its name."""

    strategies = {
        "boltzmann": Boltzmann,
        "decay_boltzmann": DecayBoltzmann,
        "epsilon": EpsilonGreedy,
        "decay_epsilon": DecayEpsilonGreedy,
        "pheromones": Pheromones,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown exploration strategy: {strategy_name}")

    if strategy_name == "pheromones":
        state_map_dict = {"tuple": lambda x: tuple(x)}

        kwargs["state_map"] = state_map_dict[kwargs["state_map"]]

    return strategies[strategy_name](*args, **kwargs)  # initialize with provided arguments
