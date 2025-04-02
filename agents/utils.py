from .democratic_dwn import DemocraticDWL, DemocraticDWL_MaxAction, DemocraticDWL_RandomMaxAction, DemocraticDWL_MaxActionMaxQValue
from .AbstractAgent import AbstractAgent
from .dwn import DWL
from .scaled_democratic import ScaledDemocraticDQN
from .democratic import DemocraticDQN
from action_scalarization import get_scalarization_method

agent_names = [
    "scaled_democratic",
    "dwl",
    "democratic",
    "democratic_dwl",
    "democratic_dwl_max_action",
    "democratic_dwl_random_max_action",
    "democratic_dwl_max_action_max_q_value",
]


def get_agent(agent_name: str, **kwargs) -> AbstractAgent:
    if kwargs.get("scalarization_method", "linear") is not None:
        kwargs["scalarization_method"] = get_scalarization_method(kwargs["scalarization_method"])

    if agent_name == "scaled_democratic":
        return ScaledDemocraticDQN(**kwargs)
    elif agent_name == "dwl":
        return DWL(**kwargs)
    elif agent_name == "democratic":
        return DemocraticDQN(**kwargs)
    elif agent_name.startswith("democratic_dwl"):
        return get_democratic_dwl_agent(agent_name, **kwargs)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}. Possible values are: {agent_names}")


def get_democratic_dwl_agent(agent_name: str, **kwargs) -> AbstractAgent:
    if agent_name == "democratic_dwl":
        return DemocraticDWL(**kwargs)
    elif agent_name == "democratic_dwl_max_action":
        return DemocraticDWL_MaxAction(**kwargs)
    elif agent_name == "democratic_dwl_random_max_action":
        return DemocraticDWL_RandomMaxAction(**kwargs)
    elif agent_name == "democratic_dwl_max_action_max_q_value":
        return DemocraticDWL_MaxActionMaxQValue(**kwargs)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}. Possible values are: {agent_names}")
