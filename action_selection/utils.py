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
        "epsilon_greedy": EpsilonGreedy,
        "decay_epsilon_greedy": DecayEpsilonGreedy,
        "pheromones": Pheromones,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown exploration strategy: {strategy_name}")

    return strategies[strategy_name](*args, **kwargs)  # Initialize with provided arguments
