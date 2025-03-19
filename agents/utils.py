from .democratic_dwn import DemocraticDWL, DemocraticDWL_MaxAction, DemocraticDWL_RandomMaxAction, DemocraticDWL_MaxActionMaxQValue
from .AbstractAgent import AbstractAgent
from .dwn import DWL
from .scaled_democratic import ScaledDemocraticDQN
from .democratic import DemocraticDQN


agent_names = [
    "scaled_democratic",
    "dwl",
    "democratic",
    "democratic_dwl",
    "democratic_dwl_max_action",
    "democratic_dwl_random_max_action",
    "democratic_dwl_max_action_max_q_value",
]


def get_agent(agent_name: str) -> AbstractAgent:
    if agent_name == "scaled_democratic":
        return ScaledDemocraticDQN
    elif agent_name == "dwl":
        return DWL
    elif agent_name == "democratic":
        return DemocraticDQN
    elif agent_name.startswith("democratic_dwl"):
        return get_democratic_dwl_agent(agent_name)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}. Possible values are: {agent_names}")


def get_democratic_dwl_agent(agent_name: str) -> AbstractAgent:
    if agent_name == "democratic_dwl":
        return DemocraticDWL
    elif agent_name == "democratic_dwl_max_action":
        return DemocraticDWL_MaxAction
    elif agent_name == "democratic_dwl_random_max_action":
        return DemocraticDWL_RandomMaxAction
    elif agent_name == "democratic_dwl_max_action_max_q_value":
        return DemocraticDWL_MaxActionMaxQValue
    else:
        raise ValueError(f"Invalid agent name: {agent_name}. Possible values are: {agent_names}")
