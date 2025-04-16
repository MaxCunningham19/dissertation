from .democratic_dwn import DemocraticDWL, ScaledDemocraticDWL
from .AbstractAgent import AbstractAgent
from .dwn import DWL
from .scaled_democratic import ScaledDemocraticDQN
from .democratic import DemocraticDQN
from action_scalarization import get_scalarization_method

agent_names = ["scaled_democratic", "dwl", "democratic", "democratic_dwl", "scaled_democratic_dwl"]


def get_agent(agent_name: str, **kwargs) -> AbstractAgent:
    if kwargs.get("scalarization", None) is not None:
        kwargs["scalarization"] = get_scalarization_method(kwargs["scalarization"])
    if agent_name == "scaled_democratic":
        return ScaledDemocraticDQN(**kwargs)
    elif agent_name == "dwl":
        return DWL(**kwargs)
    elif agent_name == "democratic":
        return DemocraticDQN(**kwargs)
    elif agent_name == "democratic_dwl":
        return DemocraticDWL(**kwargs)
    elif agent_name == "scaled_democratic_dwl":
        return ScaledDemocraticDWL(**kwargs)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}. Possible values are: {agent_names}")
